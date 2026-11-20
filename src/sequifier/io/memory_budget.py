"""Preflight and resident-storage accounting for prepared-data loading.

The budget belongs to one training rank and its loader workers. It does not cap
model/optimizer memory or the Python process's address space. Estimates reserve
decode/conversion and prefetch headroom before any dataset is materialized.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import zipfile
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from sequifier.config.metadata import validate_persisted_metadata_versions
from sequifier.helpers import PANDAS_TO_TORCH_TYPES, normalize_path


def resident_bytes(value: Any, seen: set[Any] | None = None) -> int:
    """Count CPU backing stores once, including aliased views and Python indices."""
    if seen is None:
        seen = set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            return 0
        storage = value.untyped_storage()
        identity = ("storage", storage._cdata)
        if identity in seen:
            return 0
        seen.add(identity)
        return sys.getsizeof(value) + storage.nbytes()
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(
            resident_bytes(key, seen) + resident_bytes(item, seen)
            for key, item in value.items()
            if key != "config"
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return size + sum(resident_bytes(item, seen) for item in value)
    if isinstance(value, type):
        return size
    try:
        fields = vars(value)
    except TypeError:
        return size
    return size + resident_bytes(fields, seen)


def _parquet_size(path: Path, string_width: int) -> tuple[int, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).metadata
    schema = pq.read_schema(path)
    row_width = 0
    for field in schema:
        dtype = field.type
        if pa.types.is_dictionary(dtype):
            dtype = dtype.value_type
        if pa.types.is_boolean(dtype):
            row_width += 1
        elif pa.types.is_integer(dtype) or pa.types.is_floating(dtype):
            row_width += dtype.bit_width // 8
        else:
            # Prepared tables contain numeric values plus feature-name strings.
            row_width += string_width + 8
        row_width += 1  # Validity and alignment headroom.
    decoded = metadata.num_rows * row_width
    pages = sum(
        metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups)
    )
    return max(decoded, pages), metadata.num_rows


def _source_size(
    path: Path, read_format: str, string_width: int
) -> tuple[int, int | None]:
    if read_format == "parquet":
        return _parquet_size(path, string_width)
    if read_format == "pt":
        if not zipfile.is_zipfile(path):
            raise ValueError(
                f"RAM-budget preflight requires a current ZIP-based PT payload: {path}"
            )
        with zipfile.ZipFile(path) as archive:
            return sum(item.file_size for item in archive.infolist()), None
    if read_format == "csv":
        # Streaming inspection counts quoted records correctly without loading
        # feature values into a DataFrame just to estimate its memory demand.
        rows, decoded = 0, 0
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            next(reader, None)
            for row in reader:
                rows += 1
                decoded += sum(max(8, len(item.encode("utf-8")) + 8) for item in row)
        return decoded, rows
    raise ValueError(f"Cannot plan RAM for prepared format {read_format!r}")


def _column_bytes(config: Any, column: str) -> int:
    dtype = PANDAS_TO_TORCH_TYPES[config.column_data_types[column]]
    width = torch.empty((), dtype=dtype).element_size()
    layout = config.depth_layouts.layout_for_column(column)
    return width * (layout.max_depth_length if layout is not None else 1)


def _batch_bytes(config: Any) -> int:
    context = config.window_view.context_length
    values = context * sum(
        _column_bytes(config, column)
        for column in config.input_columns + config.target_columns
    )
    values += context * (
        32 + sum(layout.max_depth_length for _, layout in config.depth_layouts.items())
    )
    values += sum(_column_bytes(config, column) for column in config.static_columns)
    for source in config.image_sources.values():
        values += context * source.frame_capacity * (3 * source.crop_size**2 + 1)
    prefix = config.preconditioning_image
    if prefix is not None:
        values += len(prefix.views) * 3 * prefix.crop_size**2
    return config.batch_size * (values + 64)


@dataclass(frozen=True)
class LoaderMemoryEstimate:
    resident: int
    working: int
    chunk_limit: int

    @property
    def total(self) -> int:
        return self.resident + self.working


def _estimate_loader(path: str, config: Any, eager: bool) -> LoaderMemoryEstimate:
    source = Path(normalize_path(path, config.project_root))
    width = config.storage_layout.window_length
    infos = None
    metadata_bytes = 0
    if source.is_dir():
        with (source / "metadata.json").open() as stream:
            metadata = json.load(stream)
        validate_persisted_metadata_versions(metadata)
        metadata_bytes = resident_bytes(metadata)
        width = metadata["window_length"]
        infos = metadata["batch_files"]
    files = (
        [(source / info["path"], int(info["samples"])) for info in infos]
        if infos is not None
        else [(source, None)]
    )
    columns = set(config.input_columns + config.target_columns)
    stored_row = width * sum(_column_bytes(config, column) for column in columns) + 48
    stored_row += width * sum(
        layout.max_depth_length for _, layout in config.depth_layouts.items()
    )
    stride = config.window_stride
    views = (
        1
        if stride is None
        else max(
            1,
            (
                width
                - config.storage_layout.max_target_offset
                - config.window_view.context_length
            )
            // stride
            + 1,
        )
    )
    string_width = (
        max((len(name.encode("utf-8")) for name in config.column_data_types), default=0)
        + 64
    )
    residents, decode_peak = [], 0
    for file, samples in files:
        decoded, rows = _source_size(file, config.read_format, string_width)
        if samples is None:
            samples = (
                math.ceil(rows / max(1, len(columns)))
                if rows is not None
                else math.ceil(decoded / max(1, stored_row))
            )
        # Window indices, curriculum order/tie groups and epoch shuffles include
        # Python lists as well as tensor indices. Reserve their expanded size.
        indices = samples * (64 + 160 * views)
        resident = samples * stored_row + indices
        residents.append(resident)
        decode_peak = max(decode_peak, 3 * decoded + 2 * resident)
    static = 0
    if config.static_features_path is not None and config.static_columns:
        static_path = Path(
            normalize_path(config.static_features_path, config.project_root)
        )
        static = 3 * _parquet_size(static_path, string_width)[0]
    image_sources = list(config.image_sources.values())
    if config.preconditioning_image is not None:
        image_sources.append(config.preconditioning_image)
    cache = 128 * max((3 * item.crop_size**2 for item in image_sources), default=0)
    workers = config.runtime_settings.num_workers
    copies = max(1, workers)
    batch = _batch_bytes(config)
    # PyTorch's worker queue has four prefetched batches per worker. Pinning
    # can overlap the queued copy; reserve both it and the consumer's batch.
    prefetch = 2 * (4 * workers + 1) * batch
    base = metadata_bytes + static
    if eager:
        resident = (sum(residents) + base) * (workers + 1)
        working = decode_peak + copies * cache + prefetch
    else:
        resident = base * (workers + 1)
        working = copies * (decode_peak + cache + 2 * batch) + prefetch
    return LoaderMemoryEstimate(resident, working, decode_peak)


@dataclass
class DataLoadingBudget:
    limit: int
    estimates: dict[str, LoaderMemoryEstimate]

    def check(self) -> None:
        required = sum(estimate.total for estimate in self.estimates.values())
        if required > self.limit:
            raise MemoryError(
                "Prepared-data loading exceeds run.execution.data_loading.max_ram_gb: "
                f"planned {required / 1e9:.3f} GB, budget {self.limit / 1e9:.3f} GB "
                "per training rank including loader workers. Use lazy folders "
                "(load_full_data_to_ram: false), smaller shards/batches, fewer "
                "workers, or increase max_ram_gb. Model/optimizer memory is additional."
            )

    def validate_dataset(self, key: str, dataset: Any, workers: int) -> None:
        estimate = self.estimates[key]
        measured = resident_bytes(dataset) * (workers + 1)
        self.estimates[key] = replace(
            estimate, resident=max(estimate.resident, measured)
        )
        self.check()
        # Plain integers survive DataLoader spawn without serializing the
        # run-wide budget or retaining references to other loaders.
        dataset.ram_chunk_limit = estimate.chunk_limit


def _estimate_indexed(path, config):
    from sequifier.io.indexed_sequences import (
        WINDOW_CHUNK_SIZE,
        IndexedView,
        WindowIndex,
    )

    view = IndexedView.load(path)
    layout = view.layout(config.window_view)
    width = layout.window_length
    index = WindowIndex(view, width)
    rows = index.count
    columns = set(config.input_columns + config.target_columns)
    row_bytes = width * (
        sum(_column_bytes(config, column) for column in columns)
        + sum(layout.max_depth_length for _, layout in config.depth_layouts.items())
    )
    workers = config.runtime_settings.num_workers
    resident = (workers + 1) * (resident_bytes(index) + index.chunk_count * 960)
    if config.static_features_path and config.static_columns:
        resident += (
            3
            * (workers + 1)
            * _parquet_size(
                Path(normalize_path(config.static_features_path, config.project_root)),
                64,
            )[0]
        )
    batch = _batch_bytes(config)
    chunk = 4 * min(rows, WINDOW_CHUNK_SIZE) * (row_bytes + 256) + 8_000_000
    chunk += min(rows, WINDOW_CHUNK_SIZE) * (64 + 8 * len(index.curriculum))
    return LoaderMemoryEstimate(
        resident,
        max(1, workers) * chunk + 2 * (4 * workers + 1) * batch,
        chunk,
    )


def plan_data_loading(config: Any) -> DataLoadingBudget:
    """Budget every cached source and the largest phase working requirements."""
    from sequifier.config.train_config import dataset_part_view
    from sequifier.evaluation.compatibility import evaluation_batch_size

    settings = config.runtime_settings
    if not math.isfinite(settings.max_ram_gb) or settings.max_ram_gb <= 0:
        raise ValueError("max_ram_gb must be finite and positive")
    budget = DataLoadingBudget(int(Decimal(str(settings.max_ram_gb)) * 10**9), {})
    phases = config.training_plan
    batch = max(
        [phase.batch_size for phase in phases]
        + [evaluation_batch_size(phase) for phase in phases],
        default=1,
    )
    strides = [
        phase.learning.window_stride
        for phase in phases
        if phase.learning.window_stride is not None
    ]
    stride = min(strides) if strides else None
    selected_parts = dict.fromkeys(
        selection for phase in phases for selection in phase.part_roles()
    )
    for name, part_name, split in selected_parts:
        part = config.dataset_training[name].parts[part_name]
        path = part.split_paths[split]
        selected = dataset_part_view(
            config, name, part_name, batch_size=batch, window_stride=stride
        )
        normalized = normalize_path(path, config.project_root)
        splits = [
            normalize_path(item, config.project_root)
            for item in part.metadata.split_paths
        ]
        layout = (
            part.metadata.layout_for_split(splits.index(normalized))
            if normalized in splits
            else selected.storage_layout
        )
        view = replace(selected, storage_layout=layout)
        eager = part.storage_form == "file" or settings.load_full_data_to_ram
        budget.estimates[f"{name}.{part_name}.{split}"] = (
            _estimate_indexed(path, view)
            if part.storage_form == "indexed"
            else _estimate_loader(path, view, eager)
        )
    budget.check()
    logger.info(
        "Prepared-data RAM plan: {:.3f} GB / {:.3f} GB per rank (including workers)",
        sum(item.total for item in budget.estimates.values()) / 1e9,
        budget.limit / 1e9,
    )
    return budget
