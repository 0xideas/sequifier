"""Ordered, bounded preprocessing for one repeated-row depth layout."""

import heapq
import json
import math
import os
import pickle
import tempfile
from collections import Counter, deque
from contextlib import closing
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Generator

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from loguru import logger

from sequifier.config.depth_layout import DepthLayoutModel, DepthLayoutRegistryModel
from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.fingerprints import file_checksum
from sequifier.helpers import PANDAS_TO_TORCH_TYPES
from sequifier.io.feature_fitting import (
    FeatureStatistics,
    FittedColumns,
    FittingRules,
    cast_input_columns,
    validate_cardinality_values,
)
from sequifier.io.pt_payload import (
    StoredTensorBatch,
    concatenate_pt_batches,
    save_pt_payload,
)
from sequifier.io.sample_order import curriculum_columns
from sequifier.io.window_placement import split_intervals


@dataclass
class RawItem:
    sequence_id: int
    position: int
    rows: list[dict]
    source: str


_MERGE_FAN_IN = 32


@lru_cache(maxsize=4)
def _parquet_file(path: str) -> pq.ParquetFile:
    return pq.ParquetFile(path)


def _file_items(
    filename: str,
    read_format: str,
    columns: list[str],
    capacity: int,
    stop_key: tuple[int, int] | None = None,
    row_groups: list[int] | None = None,
    sequence_id: int | None = None,
) -> Generator[RawItem, None, None]:
    """Yield complete items, checking order even across read batch boundaries."""

    def source_rows():
        if read_format == "csv":
            frame = pl.scan_csv(filename).select(columns)
            try:
                collect_batches = frame.collect_batches
            except AttributeError:
                reader = pl.read_csv_batched(filename, columns=columns, batch_size=1024)
                while batches := reader.next_batches(1):
                    for batch in batches:
                        yield from batch.iter_rows(named=True)
            else:
                for batch in collect_batches(chunk_size=1024):
                    yield from batch.iter_rows(named=True)
        else:
            parquet = _parquet_file(filename)
            for batch in parquet.iter_batches(
                batch_size=1024,
                row_groups=row_groups,
                columns=columns,
                use_threads=False,
            ):
                yield from batch.to_pylist()

    current_key = None
    rows = []
    with closing(source_rows()) as input_rows:
        for row in input_rows:
            key = (
                _coordinate(row["sequenceId"], "sequenceId"),
                _coordinate(row["itemPosition"], "itemPosition"),
            )
            if sequence_id is not None and key[0] != sequence_id:
                continue
            if stop_key is not None and key > stop_key:
                break
            if current_key is not None and key != current_key:
                if key < current_key:
                    raise ValueError(
                        f"{filename}: depth item coordinates must increase; "
                        f"{key} follows {current_key}"
                    )
                yield RawItem(*current_key, rows, filename)
                rows = []
            current_key = key
            rows.append(row)
            if len(rows) > capacity:
                raise ValueError(f"Item {key} exceeds depth capacity")
        if current_key is not None:
            yield RawItem(*current_key, rows, filename)


def _merge_streams(
    streams: list[Generator[RawItem, None, None]],
) -> Generator[RawItem, None, None]:
    """Merge at most one bounded batch per input stream."""
    heap = []
    try:
        for index, stream in enumerate(streams):
            item = next(stream, None)
            if item is not None:
                heapq.heappush(heap, (item.sequence_id, item.position, index, item))
        previous = None
        while heap:
            sid, pos, index, item = heapq.heappop(heap)
            if previous is not None and (sid, pos) == (
                previous.sequence_id,
                previous.position,
            ):
                raise ValueError(
                    f"Depth item {(sid, pos)} appears in both "
                    f"{previous.source} and {item.source}"
                )
            previous = item
            yield item
            following = next(streams[index], None)
            if following is not None:
                heapq.heappush(
                    heap,
                    (following.sequence_id, following.position, index, following),
                )
    finally:
        for stream in streams:
            stream.close()


def _run_items(path: Path) -> Generator[RawItem, None, None]:
    with path.open("rb") as file:
        while True:
            try:
                yield pickle.load(file)
            except EOFError:
                return


def _merged_items(
    files: list[str],
    read_format: str,
    columns: list[str],
    capacity: int,
    stop_key: tuple[int, int] | None = None,
) -> Generator[RawItem, None, None]:
    """Merge sorted files, spilling intermediate runs only beyond the fan-in limit."""
    if len(files) <= _MERGE_FAN_IN:
        yield from _merge_streams(
            [
                _file_items(path, read_format, columns, capacity, stop_key)
                for path in files
            ]
        )
        return

    with tempfile.TemporaryDirectory(prefix="depth-merge-") as directory:
        paths = []
        for start in range(0, len(files), _MERGE_FAN_IN):
            path = Path(directory) / f"first-{start}.run"
            group = files[start : start + _MERGE_FAN_IN]
            with path.open("wb") as file:
                for item in _merge_streams(
                    [
                        _file_items(source, read_format, columns, capacity, stop_key)
                        for source in group
                    ]
                ):
                    pickle.dump(item, file, protocol=pickle.HIGHEST_PROTOCOL)
            paths.append(path)
        generation = 0
        while len(paths) > _MERGE_FAN_IN:
            next_paths = []
            for start in range(0, len(paths), _MERGE_FAN_IN):
                path = Path(directory) / f"stage-{generation}-{start}.run"
                with path.open("wb") as file:
                    for item in _merge_streams(
                        [
                            _run_items(source)
                            for source in paths[start : start + _MERGE_FAN_IN]
                        ]
                    ):
                        pickle.dump(item, file, protocol=pickle.HIGHEST_PROTOCOL)
                next_paths.append(path)
            for path in paths:
                path.unlink()
            paths = next_paths
            generation += 1
        yield from _merge_streams([_run_items(path) for path in paths])


def _coordinate(value, name):
    assert isinstance(value, int) and not isinstance(
        value, bool
    ), f"{name} must be a non-null integer; found {type(value).__name__}."
    assert -(2**63) <= value < 2**63, f"{name} must fit signed Int64; found {value}."
    return value


def _observation_types(data, columns, configured):
    # Choose processing semantics without prematurely narrowing real values or
    # raw integer categories to their eventual *encoded* storage dtype.
    types = {}
    for column in columns:
        target = (configured or {}).get(column, str(data.schema[column]))
        if "float" in target.lower():
            types[column] = "Float64"
        elif data.schema[column].is_float() and "int" in target.lower():
            types[column] = "Int64"
        else:
            types[column] = str(data.schema[column])
    # The existing helper supports numeric configuration types only; string
    # categoricals are intentionally left alone.
    return {
        column: (
            types[column]
            if "float" in types[column].lower() or "int" in types[column].lower()
            else "Int64"
        )
        for column in columns
    }


@dataclass
class _RowGroupAudit:
    index: int
    first: tuple[int, int]
    last: tuple[int, int]
    first_item_rows: int
    last_item_rows: int
    counts: dict[int, int]
    first_positions: dict[int, int]


def _audit_parquet_row_group(job):
    path, index, capacity = job
    parquet = _parquet_file(path)
    counts = Counter()
    first_positions = {}
    first = last = None
    current_rows = 0
    first_item_rows = 0
    for batch in parquet.iter_batches(
        row_groups=[index],
        columns=["sequenceId", "itemPosition"],
        batch_size=8192,
        use_threads=False,
    ):
        ids, positions = (column.to_pylist() for column in batch.columns)
        for raw_sid, raw_pos in zip(ids, positions):
            key = (
                _coordinate(raw_sid, "sequenceId"),
                _coordinate(raw_pos, "itemPosition"),
            )
            if first is None:
                first = key
            if key != last:
                if last is not None:
                    if key < last:
                        raise ValueError(
                            f"{path}: depth item coordinates must increase; "
                            f"{key} follows {last}"
                        )
                    if key[0] == last[0] and key[1] != last[1] + 1:
                        raise ValueError(
                            f"itemPosition must be continuous within sequence {key[0]}"
                        )
                    if first_item_rows == 0:
                        first_item_rows = current_rows
                counts[key[0]] += 1
                first_positions.setdefault(key[0], key[1])
                current_rows = 0
            current_rows += 1
            if current_rows > capacity:
                raise ValueError(f"Item {key} exceeds depth capacity")
            last = key
    if first is None or last is None:
        raise ValueError(f"Empty Parquet row group {index} in {path}")
    return _RowGroupAudit(
        index,
        first,
        last,
        first_item_rows or current_rows,
        current_rows,
        dict(counts),
        first_positions,
    )


def _audit_parquet_groups(path, capacity, num_workers):
    """Count complete items in parallel, joining row groups at item boundaries."""
    from sequifier.preprocess import _create_preprocess_pool

    group_count = pq.ParquetFile(path).num_row_groups
    logger.info(
        "Depth audit: {} Parquet row groups across {} workers",
        group_count,
        min(num_workers, group_count),
    )
    jobs = ((path, index, capacity) for index in range(group_count))
    with _create_preprocess_pool(min(num_workers, group_count)) as pool:
        audits = list(pool.imap(_audit_parquet_row_group, jobs))
    counts = Counter()
    first_positions = {}
    groups = []
    previous = None
    trailing_rows = 0
    for audit in audits:
        if previous is not None:
            if audit.first < previous.last:
                raise ValueError(
                    f"{path}: depth item coordinates must increase; "
                    f"{audit.first} follows {previous.last}"
                )
            if audit.first[0] == previous.last[0] and audit.first != previous.last:
                if audit.first[1] != previous.last[1] + 1:
                    raise ValueError(
                        "itemPosition must be continuous within sequence "
                        f"{audit.first[0]}"
                    )
            if audit.first == previous.last:
                if trailing_rows + audit.first_item_rows > capacity:
                    raise ValueError(f"Item {audit.first} exceeds depth capacity")
                counts[audit.first[0]] -= 1
                groups[-1].append(audit.index)
            else:
                groups.append([audit.index])
        else:
            groups.append([audit.index])
        counts.update(audit.counts)
        for sid, position in audit.first_positions.items():
            first_positions.setdefault(sid, position)
        trailing_rows = (
            trailing_rows + audit.first_item_rows
            if previous is not None
            and audit.first == previous.last
            and audit.first == audit.last
            else audit.last_item_rows
        )
        previous = audit
    return dict(counts), first_positions, groups, audits


def _validated_item_rows(item, columns, shallow, layout, curriculum_columns):
    values = []
    seen = set()
    sample_positions = set()
    for raw_row in item.rows:
        if curriculum_columns:
            sample_positions.add(
                tuple(
                    _coordinate(raw_row[column], column)
                    for column in curriculum_columns
                )
            )
        child = _coordinate(
            raw_row[layout.position_column],
            layout.position_column,
        )
        if not (
            layout.position_base
            <= child
            < layout.position_base + layout.max_depth_length
        ):
            raise ValueError(f"Depth position {child} is outside layout capacity")
        slot = child - layout.position_base
        if slot in seen:
            raise ValueError(
                f"Duplicate outer/depth coordinate "
                f"{(item.sequence_id, item.position, child)}"
            )
        seen.add(slot)
        raw = {c: raw_row[c] for c in columns}
        for column, value in raw.items():
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                raise ValueError(
                    f"{column}: selected observations must be non-null and finite"
                )
        if values and any(raw[c] != values[0][1][c] for c in shallow):
            raise ValueError(
                f"Shallow values disagree for outer item "
                f"{(item.sequence_id, item.position)}"
            )
        values.append((slot, raw))
    if not layout.allow_gaps and sorted(seen) != list(range(len(seen))):
        raise ValueError(
            f"Item {(item.sequence_id, item.position)} has forbidden depth gaps"
        )
    if len(sample_positions) > 1:
        raise ValueError(
            "Curriculum columns must be identical across repeated child rows "
            f"for outer item {(item.sequence_id, item.position)}"
        )
    return (
        sorted(values),
        next(iter(sample_positions)) if sample_positions else None,
    )


@dataclass(frozen=True)
class DepthFitPlan:
    layout: DepthLayoutModel
    columns: list[str]
    shallow: list[str]
    curriculum: tuple[str, ...]
    schema: dict
    configured_types: dict | None
    rules: FittingRules
    fit_end: dict[int, int]

    def observe_items(self, items):
        features = FeatureStatistics()
        for item in items:
            rows, _ = _validated_item_rows(
                item, self.columns, self.shallow, self.layout, self.curriculum
            )
            end = self.fit_end.get(item.sequence_id)
            fitting = end is not None and item.position < end
            for name in self.columns if fitting else self.rules.cardinality:
                values = (
                    [raw[name] for _, raw in rows]
                    if name in self.layout.columns
                    else [rows[0][1][name]]
                )
                frame = pl.DataFrame({name: values}, schema={name: self.schema[name]})
                if name in self.rules.cardinality:
                    validate_cardinality_values(
                        frame, {name: self.rules.cardinality[name]}
                    )
                if fitting:
                    types = _observation_types(frame, [name], self.configured_types)
                    frame = cast_input_columns(frame, [name], types)
                    features.observe(frame, [name], self.rules)
        return features

    def fit_parquet_group(self, path, required, group):
        return self.observe_items(
            _file_items(
                path,
                "parquet",
                required,
                self.layout.max_depth_length,
                row_groups=group,
            )
        )


@dataclass
class DepthItemEncoder:
    layout_name: str
    layouts: DepthLayoutRegistryModel
    features: FittedColumns
    shallow: list[str]
    curriculum: tuple[str, ...]
    schema: dict
    configured_types: dict
    normalize: bool
    cardinality: dict

    @property
    def layout(self):
        return self.layouts[self.layout_name]

    def encode(self, item):
        rows, curriculum = _validated_item_rows(
            item, self.features.data_columns, self.shallow, self.layout, self.curriculum
        )
        slots = [slot for slot, _ in rows]
        data = pl.DataFrame([raw for _, raw in rows], schema=self.schema)
        data = self.features.transform(
            data,
            self.normalize,
            self.cardinality,
            input_types=_observation_types(
                data, self.features.data_columns, self.configured_types
            ),
        )
        capacity = self.layout.max_depth_length
        mask = np.zeros(capacity, dtype=bool)
        mask[slots] = True
        values = {}
        for column in self.features.data_columns:
            if column in self.shallow:
                values[column] = data[column][0]
            else:
                observed = data[column].to_numpy()
                dense = np.zeros(capacity, dtype=observed.dtype)
                dense[slots] = observed
                values[column] = dense
        return values, mask, curriculum

    def window(self, encoded, sid, subsequence, placement, start, pad):
        width = placement.width
        capacity = self.layout.max_depth_length
        tensors = {
            c: torch.zeros(
                (1, width) if c in self.shallow else (1, width, capacity),
                dtype=PANDAS_TO_TORCH_TYPES[self.features.col_types[c]],
            )
            for c in self.features.data_columns
        }
        mask = torch.zeros((1, width, capacity), dtype=torch.bool)
        sample_position = None
        for time in range(pad, width):
            position = start + time
            if position not in encoded:
                raise ValueError(
                    f"Missing depth item {(sid, position)} in ordered stream"
                )
            values, valid, curriculum = encoded[position]
            if sample_position is not None and sample_position != curriculum:
                raise ValueError(
                    f"Curriculum columns must agree within window {(sid, subsequence)}"
                )
            sample_position = curriculum
            mask[0, time] = torch.from_numpy(valid)
            for column, value in values.items():
                tensors[column][0, time] = torch.as_tensor(
                    value, dtype=tensors[column].dtype
                )
        batch = StoredTensorBatch(
            tensors,
            torch.tensor([sid]),
            torch.tensor([subsequence]),
            torch.tensor([start]),
            torch.tensor([pad]),
            split_start_item_positions=torch.tensor([placement.low], dtype=torch.int64),
            split_end_item_positions=torch.tensor([placement.high], dtype=torch.int64),
            depth_valid_masks={self.layout_name: mask},
            sample_positions=torch.tensor([sample_position], dtype=torch.int64)
            if sample_position is not None
            else None,
            curriculum_columns=self.curriculum,
        )
        batch.validate(self.layouts, n_classes=self.features.n_classes)
        return batch


def _render_depth_windows(
    items,
    sequences,
    plan,
    batches_per_file,
    encoder,
    scratch_root,
    completed,
    shard_path,
    record,
):
    """Stream ordered items once, retaining at most one window and one shard per split."""
    output = [[] for _ in plan.widths]
    counts = Counter()
    expected = set()
    max_width = max(plan.widths)

    def flush(split, filename):
        destination = scratch_root / filename
        temporary = destination.with_suffix(".pt.tmp")
        batch = concatenate_pt_batches(output[split])
        save_pt_payload(
            batch,
            temporary,
            layouts=encoder.layouts,
            n_classes=encoder.features.n_classes,
        )
        os.replace(temporary, destination)
        record(
            filename,
            {"size": destination.stat().st_size, "sha256": file_checksum(destination)},
        )
        output[split].clear()

    with closing(items):
        item = next(items, None)
        for sid, first, count, assigned in sequences:
            encoded = {}
            order = deque()
            for split, placement in plan.placements(first, count, assigned=assigned):
                for subsequence, (start, pad) in enumerate(placement):
                    filename = shard_path(split, counts[split] // batches_per_file)
                    expected.add(filename)
                    counts[split] += 1
                    if filename in completed:
                        continue
                    stop = start + placement.width
                    # A wider aligned window in the next split may need earlier items.
                    while item is not None and (item.sequence_id, item.position) < (
                        sid,
                        stop - max_width,
                    ):
                        item = next(items, None)
                    while (
                        item is not None
                        and item.sequence_id == sid
                        and item.position < stop
                    ):
                        encoded[item.position] = encoder.encode(item)
                        order.append(item.position)
                        if len(order) > max_width:
                            del encoded[order.popleft()]
                        item = next(items, None)
                    output[split].append(
                        encoder.window(encoded, sid, subsequence, placement, start, pad)
                    )
                    if len(output[split]) == batches_per_file:
                        flush(split, filename)
        for split, pending in enumerate(output):
            if pending:
                flush(split, shard_path(split, counts[split] // batches_per_file))
    return expected


def _render_parquet_sequence(job):
    encoder = job["encoder"]
    sequence = job["sequence"]
    items = _file_items(
        job["path"],
        "parquet",
        job["required"],
        encoder.layout.max_depth_length,
        row_groups=job["row_groups"],
        sequence_id=sequence[0],
    )

    def shard_path(split, index):
        return (
            Path(job["split_paths"][split]).stem
            + f"-0-{job['sequence_index']:012d}{index:012d}.pt"
        )

    records = {}
    expected = _render_depth_windows(
        items,
        [sequence],
        job["plan"],
        job["batches_per_file"],
        encoder,
        job["scratch_root"],
        job["completed"],
        shard_path,
        records.__setitem__,
    )
    return expected, records


def preprocess_depth(owner, selected_columns):
    from sequifier.preprocess import (
        _between_sequence_assignments,
        _folder_input_files,
        _validate_cardinality_columns,
        _validate_declared_column_roles,
        assign_sequence_to_split,
        load_precomputed_id_maps,
    )

    config = owner.config
    if config.max_rows is not None and config.max_rows <= 0:
        raise ValueError("Depth max_rows must be a positive logical item count")
    layouts = config.depth_layouts
    name, layout = next(iter(layouts.items()))
    files = (
        _folder_input_files(
            config.preprocessing_data_path,
            config.read_format,
            config.static_data_path,
        )
        if Path(config.preprocessing_data_path).is_dir()
        else [config.preprocessing_data_path]
    )
    configured_curriculum_columns = curriculum_columns(config.curriculum_column)
    reserved_columns = {
        "sequenceId",
        "itemPosition",
        layout.position_column,
        *configured_curriculum_columns,
        *config.indexed_passthrough,
    }
    columns = [c for c in selected_columns or [] if c not in reserved_columns]
    _validate_declared_column_roles(
        columns, config.categorical_columns, config.real_columns
    )
    scratch_root = Path(config.project_root) / "data" / owner.target_dir
    schema_types = None
    sample_position_presence = None
    required: list[str] = []
    for filename in files:
        scan = (
            pl.scan_csv(filename)
            if config.read_format == "csv"
            else pl.scan_parquet(filename)
        )
        schema = scan.collect_schema()
        if layout.position_column not in schema:
            raise ValueError(
                f"{filename} is missing depth position column "
                f"{layout.position_column!r}"
            )
        if not columns:
            columns = [c for c in schema if c not in reserved_columns]
        current_has_positions = bool(configured_curriculum_columns) and all(
            column in schema for column in configured_curriculum_columns
        )
        if configured_curriculum_columns and not current_has_positions:
            missing = set(configured_curriculum_columns) - set(schema)
            raise ValueError(
                f"{filename} is missing curriculum columns {sorted(missing)}"
            )
        if sample_position_presence is None:
            sample_position_presence = current_has_positions
        elif sample_position_presence != current_has_positions:
            raise ValueError(
                "The curriculum column must be present in every input file"
            )
        for curriculum_column in configured_curriculum_columns:
            if not schema[curriculum_column].is_integer():
                raise ValueError(
                    f"{curriculum_column} must have an integer dtype in "
                    f"{filename}; found {schema[curriculum_column]}."
                )
        if not set(layout.columns) <= set(columns):
            raise ValueError("Selected features must include every depth layout column")
        required = list(
            dict.fromkeys(
                [
                    "sequenceId",
                    "itemPosition",
                    layout.position_column,
                    *configured_curriculum_columns,
                    *columns,
                    *config.indexed_passthrough,
                ]
            )
        )
        missing = set(required) - set(schema)
        if missing:
            raise ValueError(f"{filename}: missing columns {sorted(missing)}")
        for coordinate in ("sequenceId", "itemPosition"):
            dtype = schema[coordinate]
            assert (
                dtype.is_integer()
            ), f"{coordinate} must have an integer dtype in {filename}; found {dtype}."
        types = {c: schema[c] for c in columns}
        if schema_types is not None and types != schema_types:
            raise ValueError(
                "Depth source files must use consistent selected feature dtypes"
            )
        schema_types = types
    owner.has_sample_positions = bool(sample_position_presence)
    if not schema_types:
        raise ValueError("Depth preprocessing source is empty")

    # The first pass audits every source item, including the part after max_rows.
    # Distinct keys in each file must increase; the merge rejects cross-file duplicates.
    source_snapshot = tuple(
        (path, Path(path).stat().st_size, Path(path).stat().st_mtime_ns)
        for path in files
    )

    def check_sources():
        current_files = (
            _folder_input_files(
                config.preprocessing_data_path,
                config.read_format,
                config.static_data_path,
            )
            if Path(config.preprocessing_data_path).is_dir()
            else [config.preprocessing_data_path]
        )
        if current_files != files:
            raise ValueError("Depth source file set changed during preprocessing")
        current = tuple(
            (path, Path(path).stat().st_size, Path(path).stat().st_mtime_ns)
            for path in files
        )
        if current != source_snapshot:
            raise ValueError("Depth source files changed during preprocessing")

    sequence_counts = Counter()
    first_positions = {}
    selected_count = 0
    previous = None
    parallel_groups = None
    parallel_audits = None
    if (
        config.read_format == "parquet"
        and len(files) == 1
        and config.max_rows is None
        and owner.num_workers > 1
        and pq.ParquetFile(files[0]).num_row_groups > 1
    ):
        (
            sequence_counts,
            first_positions,
            parallel_groups,
            parallel_audits,
        ) = _audit_parquet_groups(files[0], layout.max_depth_length, owner.num_workers)
        selected_count = sum(sequence_counts.values())
        previous = parallel_audits[-1].last
    else:
        for item in _merged_items(
            files,
            config.read_format,
            ["sequenceId", "itemPosition"],
            layout.max_depth_length,
        ):
            if config.max_rows is not None and selected_count >= config.max_rows:
                continue
            sid, pos = item.sequence_id, item.position
            if previous is not None and sid == previous[0] and pos != previous[1] + 1:
                raise ValueError(
                    f"itemPosition must be continuous within sequence {sid}"
                )
            sequence_counts[sid] += 1
            first_positions.setdefault(sid, pos)
            selected_count += 1
            previous = (sid, pos)
    if selected_count == 0:
        raise ValueError("No logical items selected for depth preprocessing")
    selected_stop = previous
    check_sources()

    def selected_items():
        check_sources()
        stream = _merged_items(
            files, config.read_format, required, layout.max_depth_length, selected_stop
        )
        try:
            for _ in range(selected_count):
                item = next(stream, None)
                if item is None:
                    raise ValueError("Depth source ended during preprocessing")
                yield item
        finally:
            stream.close()
            check_sources()

    shallow = [c for c in columns if c not in layout.columns]

    precomputed = load_precomputed_id_maps(
        config.project_root, columns, config.use_precomputed_maps
    )
    sequence_counts = dict(sequence_counts)
    assignments = (
        _between_sequence_assignments(
            sequence_counts,
            owner.window_plan,
            config.seed,
            owner._sequence_assignments(sequence_counts),
        )
        if config.split_method == "between_sequence"
        else {}
    )
    first_positions = dict(first_positions)
    existing = owner._frozen_metadata
    fitted = None
    if existing is not None:
        if (
            DepthLayoutRegistryModel.model_validate(existing.get("depth_layouts", {}))
            != layouts
        ):
            raise ValueError(
                "metadata_config_path requires identical complete depth layouts"
            )
        if existing.get("normalize_real_columns", True) != owner.normalize_real_columns:
            raise ValueError(
                "Metadata normalization policy does not match preprocessing"
            )
        fitted = owner._frozen_temporal_features(data_columns=columns)
    _validate_cardinality_columns(owner.cardinality_config, columns, precomputed)
    rules = FittingRules(
        precomputed,
        owner.cardinality_config,
        config.categorical_columns,
        config.real_columns,
    )
    fit_end = {}
    if existing is None:
        for sid, count in sequence_counts.items():
            if not config.normalize_on_all_data:
                if config.split_method == "between_sequence":
                    split = assignments.get(
                        sid,
                        assign_sequence_to_split(sid, config.split_ratios, config.seed),
                    )
                    if split != 0:
                        continue
                else:
                    _, count = split_intervals(count, config.split_ratios)[0]
            fit_end[sid] = first_positions[sid] + count
    fit_plan = DepthFitPlan(
        layout,
        columns,
        shallow,
        configured_curriculum_columns,
        schema_types,
        owner.column_data_types,
        rules,
        fit_end,
    )
    if parallel_groups is not None and existing is None:
        from sequifier.preprocess import _create_preprocess_pool

        workers = min(owner.num_workers, len(parallel_groups))
        logger.info(
            "Depth metadata fit: {} source partitions across {} workers",
            len(parallel_groups),
            workers,
        )
        features = FeatureStatistics()
        with _create_preprocess_pool(workers) as pool:
            worker = partial(fit_plan.fit_parquet_group, files[0], required)
            for summary in pool.imap(worker, parallel_groups):
                features.merge(summary)
        check_sources()
    else:
        features = fit_plan.observe_items(selected_items())
    if fitted is None:
        id_maps, stats = features.finish(rules)
        fitted = FittedColumns.resolve(
            columns, id_maps, stats, schema_types, types=owner.column_data_types
        )
        if set(columns) - set(fitted.col_types):
            raise ValueError("Column dtypes must cover selected features")
    encoder = DepthItemEncoder(
        layout_name=name,
        layouts=layouts,
        features=fitted,
        shallow=shallow,
        curriculum=configured_curriculum_columns,
        schema=schema_types,
        configured_types=owner.column_data_types,
        normalize=owner.normalize_real_columns,
        cardinality=owner.cardinality_config,
    )

    def physical_item(item):
        domain = {column: item.rows[0][column] for column in shallow}
        for column in layout.columns:
            dense = [None] * layout.max_depth_length
            for row in item.rows:
                dense[row[layout.position_column] - layout.position_base] = row[column]
            domain[column] = dense
        return domain

    if owner.output.indexed:
        from itertools import groupby

        from sequifier.config.depth_layout import depth_mask_metadata_key
        from sequifier.io.sample_order import curriculum_storage_column
        from sequifier.io.sequence import DOMAIN_OBSERVATIONS

        owner._export_metadata(fitted)
        for sid, items in groupby(selected_items(), key=lambda item: item.sequence_id):
            records = []
            for item in items:
                values, mask, curriculum = encoder.encode(item)
                record = {
                    "sequenceId": sid,
                    "itemPosition": item.position,
                    depth_mask_metadata_key(name): mask.tolist(),
                    **{
                        column: value.tolist()
                        if isinstance(value, np.ndarray)
                        else value
                        for column, value in values.items()
                    },
                }
                record[DOMAIN_OBSERVATIONS] = physical_item(item)
                for column in config.indexed_passthrough:
                    value = item.rows[0][column]
                    if any(row[column] != value for row in item.rows[1:]):
                        raise ValueError(
                            f"{column}: passthrough values disagree for outer item {(sid, item.position)}"
                        )
                    record[column] = value
                if curriculum is not None:
                    record.update(
                        (curriculum_storage_column(column), value)
                        for column, value in zip(
                            configured_curriculum_columns, curriculum
                        )
                    )
                records.append(record)
            owner.output.write_sequences(pl.DataFrame(records), assignments)
        return

    owner._write_or_validate_resume_manifest(selected_columns, "pt", fitted)
    # A shard is reusable only after both its payload and its ledger entry have
    # been committed. The ledger also ties completed shards to this source set.
    ledger_path = scratch_root / "depth-shards.json"
    source_identity = [list(entry) for entry in source_snapshot]
    parallel_render = parallel_groups is not None
    ledger_versions = {
        "serial": ContractVersions.SERIAL_DEPTH_LEDGER,
        "parallel": ContractVersions.PARALLEL_DEPTH_LEDGER,
    }
    if not config.reprocess and ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        mode = ledger.get("mode") if isinstance(ledger, dict) else None
        if not isinstance(mode, str) or mode not in ledger_versions:
            raise ValueError("Depth resume ledger requires serial or parallel mode")
        require_contract_version(
            "depth ledger", ledger.get("version"), ledger_versions[mode]
        )
        if ledger.get("sources") != source_identity:
            raise ValueError(
                "Depth resume source files differ from the completed shards"
            )
        if mode == "parallel" and not parallel_render:
            raise ValueError(
                "Depth parallel resume requires the original Parquet row groups "
                "and num_workers > 1"
            )
        parallel_render = mode == "parallel"
    else:
        mode = "parallel" if parallel_render else "serial"
        ledger = {
            "mode": mode,
            "version": ledger_versions[mode],
            "sources": source_identity,
            "shards": {},
        }

    from itertools import islice

    from sequifier.io.preprocessing_output import store_source_rows
    from sequifier.io.sequence import DOMAIN_OBSERVATIONS

    items = iter(selected_items())
    for index, batch in enumerate(iter(lambda: list(islice(items, 1024)), [])):
        rows = pl.DataFrame(
            [
                {
                    "sequenceId": item.sequence_id,
                    "itemPosition": item.position,
                    DOMAIN_OBSERVATIONS: physical_item(item),
                }
                for item in batch
            ],
            infer_schema_length=None,
        )
        store_source_rows(scratch_root, str(index), rows)
    owner._export_metadata(fitted)

    def shard_path(split, index):
        return Path(owner.split_paths[split]).stem + f"-0-{index:08d}.pt"

    def shard_location(split, filename):
        return Path(owner.split_paths[split]).with_suffix("") / filename

    completed = set()
    for split in range(len(owner.split_paths)):
        for filename, record in ledger["shards"].items():
            if not filename.startswith(Path(owner.split_paths[split]).stem + "-0-"):
                continue
            scratch = scratch_root / filename
            final = shard_location(split, filename)
            valid_scratch = (
                scratch.is_file()
                and scratch.stat().st_size == record["size"]
                and file_checksum(scratch) == record["sha256"]
            )
            valid_final = (
                final.is_file()
                and final.stat().st_size == record["size"]
                and file_checksum(final) == record["sha256"]
            )
            if valid_scratch or valid_final:
                completed.add(filename)
            if valid_final and scratch.is_file() and not valid_scratch:
                scratch.unlink()

    def commit_ledger():
        temporary = ledger_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(ledger, sort_keys=True))
        os.replace(temporary, ledger_path)

    if not ledger_path.exists():
        commit_ledger()

    sequences = (
        (
            sid,
            first_positions[sid],
            count,
            assignments.get(
                sid, assign_sequence_to_split(sid, config.split_ratios, config.seed)
            )
            if config.split_method == "between_sequence"
            else None,
        )
        for sid, count in sequence_counts.items()
    )

    if parallel_render:
        from sequifier.preprocess import _create_preprocess_pool

        row_group_audits = parallel_audits
        assert row_group_audits is not None
        check_sources()
        # A worker can be interrupted after replacing a payload but before its
        # result reaches the parent ledger. Such files have no committed record.
        for stale in scratch_root.glob("*.pt"):
            if stale.name not in completed and any(
                stale.name.startswith(Path(path).stem + "-0-")
                for path in owner.split_paths
            ):
                stale.unlink()
        expected_shards = set()

        def render_jobs():
            for index, sequence in enumerate(sequences):
                sid = sequence[0]
                yield {
                    "path": files[0],
                    "row_groups": [
                        audit.index for audit in row_group_audits if sid in audit.counts
                    ],
                    "required": required,
                    "sequence": sequence,
                    "sequence_index": index,
                    "encoder": encoder,
                    "split_paths": owner.split_paths,
                    "scratch_root": scratch_root,
                    "completed": completed,
                    "plan": owner.window_plan,
                    "batches_per_file": config.batches_per_file,
                }

        logger.info(
            "Depth windows: {} sequences across {} workers",
            len(sequence_counts),
            min(owner.num_workers, len(sequence_counts)),
        )
        with _create_preprocess_pool(
            min(owner.num_workers, len(sequence_counts))
        ) as pool:
            for names, records in pool.imap_unordered(
                _render_parquet_sequence, render_jobs()
            ):
                expected_shards.update(names)
                ledger["shards"].update(records)
                commit_ledger()
        check_sources()
    else:

        def record_shard(filename, record):
            ledger["shards"][filename] = record
            commit_ledger()

        expected_shards = _render_depth_windows(
            selected_items(),
            sequences,
            owner.window_plan,
            config.batches_per_file,
            encoder,
            scratch_root,
            completed,
            shard_path,
            record_shard,
        )
    for stale in scratch_root.glob("*.pt"):
        if stale.name not in expected_shards:
            raise ValueError(
                f"Depth temp folder contains stale output {stale}; "
                "use a fresh output location"
            )
    for split, split_path in enumerate(owner.split_paths):
        folder = Path(split_path).with_suffix("")
        if folder.exists():
            for stale in folder.glob("*.pt"):
                if stale.name not in expected_shards:
                    suffix = stale.stem.removeprefix(f"{folder.name}-0-")
                    if config.reprocess and suffix != stale.stem and suffix.isdigit():
                        stale.unlink()
                        Path(f"{stale}.metadata.json").unlink(missing_ok=True)
                        continue
                    raise ValueError(
                        f"Existing split folder contains stale output {stale}; "
                        "use a fresh output location"
                    )
