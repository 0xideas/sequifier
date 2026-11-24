"""Persistent item storage and coordinate-only views of sequence windows."""

import hashlib
import json
import os
from bisect import bisect_right
from dataclasses import dataclass
from functools import cached_property
from itertools import groupby, islice
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from typing import Literal

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from sequifier.config.depth_layout import depth_mask_metadata_key
from sequifier.config.metadata import DatasetMetadata, load_dataset_metadata
from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.fingerprints import file_checksum
from sequifier.helpers import PANDAS_TO_TORCH_TYPES, StoredWindowLayout
from sequifier.io.atomic import atomic_path
from sequifier.io.preprocessing import normalization_statistics
from sequifier.io.pt_payload import StoredTensorBatch
from sequifier.io.sequence import ordered_sequence_rows, validate_sequence_ids
from sequifier.io.window_placement import (
    PredictionAlignment,
    WindowPlacement,
    window_count,
)

DATASET_FORMAT = "sequifier.indexed-sequences"
VIEW_FORMAT = "sequifier.sequence-view"
ROW_GROUP_SIZE = 8192
WINDOW_CHUNK_SIZE = 256
PREFIX_IMAGE = "__preconditioning_image"
WINDOW_INDEX_SCHEMA = pa.schema(
    [
        *(
            (name, pa.int64())
            for name in ("sequence", "id", "start", "pad", "low", "high")
        ),
        ("curriculum", pa.list_(pa.int64())),
    ]
)


def write_json(path, values):
    path.write_text(json.dumps(values, sort_keys=True), encoding="utf-8")


def read_contract(path, format_name):
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or values.get("format") != format_name:
        raise ValueError(f"{path}: expected {format_name}")
    require_contract_version(
        format_name, values.get("version"), ContractVersions.INDEXED_SEQUENCES
    )
    return values


def is_indexed_view(path):
    if not path or Path(path).suffix.lower() != ".json":
        return False
    with Path(path).open(encoding="utf-8") as stream:
        values = json.load(stream)
    return isinstance(values, dict) and values.get("format") == VIEW_FORMAT


def prepared_split_form(path: Path) -> Literal["file", "folder", "indexed"]:
    if path.is_dir():
        return "folder"
    if not path.is_file():
        raise ValueError(f"Training data path {str(path)!r} does not exist")
    return "indexed" if is_indexed_view(path) else "file"


def _index_rows(sequence, placement, start_index=0):
    for index, (start, pad) in enumerate(placement, start_index):
        yield {
            "sequence": sequence,
            "id": index,
            "start": start,
            "pad": pad,
            "low": placement.low,
            "high": placement.high,
            "curriculum": [],
        }


class SplitIndexWriter:
    def __init__(self, path):
        self.path = path
        self.ranges = []
        self.count = 0
        self.buffer = []
        self.writer = pq.ParquetWriter(path, WINDOW_INDEX_SCHEMA)

    def add(self, sequence_id, placement, samples):
        self.ranges.append((sequence_id, placement.low, placement.high))
        for row in _index_rows(sequence_id, placement):
            if samples is not None:
                row["curriculum"] = samples[row["id"]].tolist()
            self.buffer.append(row)
            self.count += 1
            if len(self.buffer) == WINDOW_CHUNK_SIZE:
                self.flush()

    def flush(self):
        if self.buffer:
            self.writer.write_table(
                pa.Table.from_pylist(self.buffer, schema=WINDOW_INDEX_SCHEMA)
            )
            self.buffer.clear()

    def close(self):
        self.writer.close()

    def finish(self):
        self.flush()
        self.close()


class IndexedSequenceWriter:
    """Commit one ordered item store and its split views as a directory."""

    def __init__(self, output_path, *, workspace=None):
        self.destination = Path(output_path).absolute()
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        if self.destination.exists():
            raise FileExistsError(self.destination)
        self.workspace = workspace
        if workspace is None:
            self.root = Path(
                mkdtemp(
                    prefix=f".{self.destination.name}-", dir=self.destination.parent
                )
            )
        else:
            self.root = workspace / "publication"
            self.root.mkdir()
        self.files = []
        self.sequences = {}
        self.views = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.root.exists():
            for view in self.views.values():
                view.close()
            rmtree(self.root)

    def item_checkpoint(self, sequence_id):
        if self.workspace is None:
            raise ValueError("Item checkpoints require a preprocessing workspace")
        return self.workspace / f"items-{sequence_id}.parquet"

    def add(self, items, windows, *, episode_id=None, group_id=None, content_key=None):
        items = ordered_sequence_rows(items)
        sequence_ids = items["sequenceId"].unique()
        if (
            len(sequence_ids) != 1
            or not sequence_ids.dtype.is_integer()
            or sequence_ids.null_count()
        ):
            raise ValueError("Indexed sequence writes require one nonempty sequence")
        sequence_id = int(sequence_ids.item())
        if sequence_id in self.sequences:
            raise ValueError(f"Duplicate indexed sequence {sequence_id}")
        positions = items["itemPosition"]
        path = self.root / f"items-{len(self.sequences):08d}.parquet"
        if self.workspace is None:
            items.write_parquet(path, row_group_size=ROW_GROUP_SIZE)
        else:
            checkpoint = self.item_checkpoint(sequence_id)
            if not checkpoint.exists():
                with atomic_path(checkpoint) as temporary:
                    items.write_parquet(temporary, row_group_size=ROW_GROUP_SIZE)
            os.link(checkpoint, path)
        self.files.append(path)
        offsets = [0]
        with pq.ParquetFile(path) as file:
            for index in range(file.num_row_groups):
                offsets.append(offsets[-1] + file.metadata.row_group(index).num_rows)
        self.sequences[sequence_id] = {
            "path": path.name,
            "offsets": offsets,
            "position": int(positions[0]),
            "length": len(positions),
            "episode_id": episode_id,
            "group_id": group_id,
            "content_key": content_key,
        }
        for split, (placement, samples) in windows.items():
            name = str(split)
            if name not in self.views:
                self.views[name] = SplitIndexWriter(
                    self.root / f"{name}-windows.parquet"
                )
            self.views[name].add(sequence_id, placement, samples)

    def split_sequence_ids(self, split):
        view = self.views.get(str(split))
        return {sequence for sequence, _, _ in view.ranges} if view else set()

    def write_static(self, static):
        path = self.root / "static.parquet"
        static.write_parquet(path)
        self.files.append(path)

    def publish(
        self, metadata, *, feature_columns, original_metadata=None, provenance=None
    ):
        version = ContractVersions.INDEXED_SEQUENCES
        metadata = dict(metadata)
        columns = set(feature_columns) | set(metadata.get("static_columns", ()))
        for field in (
            "column_data_types",
            "id_maps",
            "n_classes",
            "cardinality_config",
            "selected_columns_statistics",
        ):
            metadata[field] = {
                name: value
                for name, value in metadata.get(field, {}).items()
                if name in columns
            }
        values = DatasetMetadata.model_validate(metadata)
        if not values.indexed_views or values.split_paths != [
            f"{name}.split.json" for name in values.indexed_views
        ]:
            raise ValueError("Indexed split paths disagree with declared views")
        if set(self.views) - set(values.indexed_views):
            raise ValueError("Indexed windows exist for undeclared views")
        if not self.sequences:
            raise ValueError("Indexed output requires at least one sequence")
        metadata_path = self.root / "metadata.json"
        write_json(metadata_path, values.model_dump(mode="json"))
        self.files.append(metadata_path)
        if original_metadata is None:
            original_path = metadata_path
        else:
            original_path = self.root / "original-metadata.json"
            if Path(original_metadata).resolve() != original_path.resolve():
                original_path.write_bytes(Path(original_metadata).read_bytes())
            self.files.append(original_path)
        for name in values.indexed_views:
            path = self.root / f"{name}.split.json"
            index = self.views.get(str(name))
            if index is not None:
                index.finish()
                self.files.append(index.path)
            write_json(
                path,
                {
                    "format": VIEW_FORMAT,
                    "version": version,
                    "dataset": "dataset.json",
                    "view": name,
                    "windows": index.path.name if index else None,
                    "window_count": index.count if index else 0,
                    "ranges": index.ranges if index else [],
                },
            )
        self.files.extend(self.root.glob("*.split.json"))
        manifest_path = self.root / "dataset.json"
        manifest = {
            "format": DATASET_FORMAT,
            "version": version,
            "metadata": metadata_path.name,
            "original_metadata": original_path.name,
            "feature_contract": feature_contract_id(metadata_path),
            "sequences": self.sequences,
            "checksums": {
                str(path.relative_to(self.root)): file_checksum(path)
                for path in self.files
            },
        }
        if provenance is not None:
            manifest["provenance"] = provenance
        write_json(manifest_path, manifest)
        view = IndexedView.load(self.root / values.split_paths[0])
        view.validate_resources()
        view.validate_item_columns(_stored_feature_columns(values))
        os.rename(self.root, self.destination)


@dataclass
class IndexedView:
    path: Path
    manifest_path: Path
    view: dict
    dataset: dict

    @classmethod
    def load(cls, path):
        path = Path(path).absolute()
        view = read_contract(path, VIEW_FORMAT)
        manifest = path.parent / view["dataset"]
        return cls(path, manifest, view, read_contract(manifest, DATASET_FORMAT))

    @property
    def root(self):
        return self.manifest_path.parent

    @property
    def metadata_path(self):
        return self.root / self.dataset["metadata"]

    @property
    def original_metadata_path(self):
        return self.root / self.dataset["original_metadata"]

    def dependencies(self):
        return [
            self.path,
            self.manifest_path,
            *(self.root / name for name in self.dataset["checksums"]),
        ]

    def verify(self, *, include_media=False):
        self.validate_resources()
        if feature_contract_id(self.metadata_path) != self.dataset["feature_contract"]:
            raise ValueError(
                f"{self.path}: indexed feature contract disagrees with metadata"
            )
        dependencies = {
            path: file_checksum(path) for path in dict.fromkeys(self.dependencies())
        }
        for name, expected in self.dataset["checksums"].items():
            path = self.root / name
            if dependencies[path] != expected:
                raise ValueError(f"Indexed dataset checksum disagrees with {path}")
        if not include_media:
            return dependencies
        from sequifier.io.images import media_files

        prefix = self._metadata.preconditioning_image
        locations = [source.path for source in self._metadata.image_sources.values()]
        if prefix and prefix.paths:
            locations.extend(prefix.paths.values())
        for location in filter(None, locations):
            for path in media_files(location, str(self.root)):
                dependencies[path] = file_checksum(path)
        return dependencies

    @cached_property
    def _metadata(self):
        return load_dataset_metadata(str(self.metadata_path))

    def metadata(self):
        return self._metadata.model_copy(deep=True)

    @property
    def static_path(self):
        return self._metadata.static_features_path

    def image_store(self, sources=None, prefix=None, *, project_root=None):
        from sequifier.io.images import ImageStore

        if sources is None:
            sources = self._metadata.image_sources
            prefix = self._metadata.preconditioning_image
        if not sources and prefix is None:
            return None
        if self._metadata.indexed_inline_media:
            if any(source.path for source in sources.values()) or (
                prefix is not None and prefix.paths is not None
            ):
                raise ValueError(
                    "Indexed inline media cannot bind external image paths"
                )
            return IndexedImageStore(self, sources, prefix)
        return ImageStore(sources, prefix, project_root or str(self.root))

    def validate_resources(self):
        for sequence in self.dataset["sequences"]:
            if str(int(sequence)) != sequence or not -(2**63) <= int(sequence) < 2**63:
                raise ValueError(
                    f"{self.path}: invalid integer sequenceId {sequence!r}"
                )
        if any(
            (isinstance(sequence, int) is False)
            for sequence, _, _ in self.view["ranges"]
        ):
            raise ValueError(f"{self.path}: split ranges require integer sequence IDs")
        metadata = self._metadata
        normalization_statistics(metadata.selected_columns_statistics)
        for column, mapping in metadata.id_maps.items():
            count = metadata.n_classes.get(column, 0)
            if column in metadata.column_data_types and any(
                not 0 <= code < count for code in mapping.values()
            ):
                raise ValueError(
                    f"{self.path}: {column} has IDs outside its vocabulary"
                )

        resources = {self.path, self.metadata_path, self.original_metadata_path}
        resources.update(Path(path) for path in metadata.split_paths)
        resources.update(
            self.root / row["path"] for row in self.dataset["sequences"].values()
        )
        if windows := self.view["windows"]:
            resources.add(self.root / windows)
        for source in metadata.image_sources.values():
            indexes = [source.time_index_path, *source.video_index_paths.values()]
            if metadata.indexed_inline_media == bool(source.path) or (
                metadata.indexed_inline_media and any(indexes)
            ):
                raise ValueError(
                    f"{self.path}: image resources disagree with the storage mode"
                )
            resources.update(Path(path) for path in indexes if path)
        prefix = metadata.preconditioning_image
        if prefix and metadata.indexed_inline_media == bool(prefix.paths):
            raise ValueError(
                f"{self.path}: preconditioning images disagree with the storage mode"
            )
        if self.static_path:
            resources.add(Path(self.static_path))
        if missing := resources - {
            self.root / name for name in self.dataset["checksums"]
        }:
            raise ValueError(
                f"{self.path}: resources missing from checksums: {sorted(missing)}"
            )

        columns = set(metadata.static_columns)
        if missing := columns - metadata.column_data_types.keys():
            raise ValueError(
                f"{self.path}: static features lack dtypes: {sorted(missing)}"
            )
        if prefix and metadata.indexed_inline_media:
            columns.add(PREFIX_IMAGE)
        if not self.static_path:
            if columns:
                raise ValueError(f"{self.path}: static features require a static table")
            return
        static = pl.scan_parquet(self.static_path)
        if missing := (columns | {"sequenceId"}) - set(static.collect_schema().names()):
            raise ValueError(
                f"{self.path}: static table lacks columns: {sorted(missing)}"
            )
        ids = static.select("sequenceId").collect()["sequenceId"]
        if not ids.dtype.is_integer() or ids.null_count() or ids.n_unique() != len(ids):
            raise ValueError(
                f"{self.path}: static table requires unique integer sequence IDs"
            )
        if missing := set(map(int, self.dataset["sequences"])) - set(ids):
            raise ValueError(
                f"{self.path}: static table lacks sequence IDs: {sorted(missing)[:10]}"
            )

    def validate_item_columns(self, required, *, struct_fields=None):
        required = set(required)
        metadata = self._metadata
        masks = {depth_mask_metadata_key(name) for name in metadata.depth_layouts.root}
        coordinates = {"sequenceId", "itemPosition"}
        for sequence, record in self.dataset["sequences"].items():
            location = f"{self.path}: sequence {sequence}"
            schema = pl.scan_parquet(self.root / record["path"]).collect_schema()
            if missing := required - set(schema.names()):
                raise ValueError(f"{location} lacks stored features: {sorted(missing)}")
            for name, fields in (struct_fields or {}).items():
                dtype = schema[name]
                if not isinstance(dtype, pl.Struct) or set(fields) - set(
                    dtype.to_schema()
                ):
                    raise ValueError(f"{location}: {name} lacks required record fields")
            for column in required:
                dtype = schema[column]
                if column in masks or metadata.depth_layouts.is_deep_column(column):
                    if not isinstance(dtype, (pl.List, pl.Array)):
                        raise ValueError(f"{location} needs a depth array for {column}")
                    dtype = dtype.inner
                declared = metadata.column_data_types.get(column, "").lower()
                if (
                    (column in masks and dtype != pl.Boolean)
                    or (
                        ("int" in declared or column in coordinates)
                        and not dtype.is_integer()
                    )
                    or ("float" in declared and not dtype.is_float())
                ):
                    raise ValueError(
                        f"{location} has an incompatible dtype for {column}: {dtype}"
                    )

    def split_bounds(self):
        return {sequence: (low, high) for sequence, low, high in self.view["ranges"]}

    @property
    def is_fitting_split(self) -> bool:
        return (
            self.view["view"] != "predict"
            and int(self.view["view"]) in self._metadata.fit_split_indices
        )

    @cached_property
    def fitted_sequences(self) -> frozenset[int]:
        return frozenset(
            sequence
            for split in self._metadata.fit_split_indices
            for sequence in IndexedView.load(
                self._metadata.split_paths[split]
            ).split_bounds()
        )

    def selection_keys(self):
        keys = {
            "sequence": set(),
            "episode_id": set(),
            "group_id": set(),
            "content_key": set(),
        }
        for sequence in self.split_bounds():
            keys["sequence"].add((self.manifest_path.resolve(), sequence))
            record = self.dataset["sequences"][str(sequence)]
            for field in ("episode_id", "group_id", "content_key"):
                if record.get(field) is not None:
                    keys[field].add(record[field])
        return keys

    def layout(self, window_view):
        metadata = self.metadata()
        index = int(self.view["view"])
        base = metadata.layout_for_split(index)
        offset = max(base.max_target_offset, window_view.target_offset)
        width = window_view.context_length + offset
        alignment = PredictionAlignment.for_split(metadata, index)
        if alignment:
            width = alignment.required_width(window_view.context_length, offset)
        return StoredWindowLayout(max(width, base.window_length), offset, base.version)

    def prediction_entries(self, context, offset, stride, right_edge, forecast):
        """Build prediction coordinates for a dataset without training splits."""
        for sequence, record in sorted(
            self.dataset["sequences"].items(), key=lambda entry: int(entry[0])
        ):
            first = record["position"]
            end = first + record["length"]
            latest = end - context
            if right_edge:
                starts = (latest,)
            else:
                latest -= offset
                if latest < first:
                    starts = (latest,)
                else:
                    starts = range(
                        first + (latest - first) % stride, latest + 1, stride
                    )
            for index, start in enumerate(starts):
                yield {
                    "sequence": int(sequence),
                    "id": index,
                    "start": start,
                    "pad": min(context, max(0, first - start)),
                    "low": end if right_edge and forecast else first,
                    "high": end + offset if right_edge and forecast else end,
                    "curriculum": [],
                }

    def read(self, sequence, start=None, end=None, columns=None) -> pl.DataFrame:
        record = self.dataset["sequences"][str(sequence)]
        first = record["position"]
        low = 0 if start is None else max(0, start - first)
        high = record["length"] if end is None else min(record["length"], end - first)
        if low >= high:
            raise ValueError(f"Empty indexed read for sequence {sequence}")
        with pq.ParquetFile(self.root / record["path"]) as file:
            offsets = record["offsets"]
            first_group = bisect_right(offsets, low) - 1
            stop_group = bisect_right(offsets, high - 1)
            groups = range(first_group, stop_group)
            table = file.read_row_groups(list(groups), columns=columns)
            frame = pl.DataFrame(table.slice(low - offsets[first_group], high - low))
            if "sequenceId" in frame.columns:
                validate_sequence_ids(frame)
            return frame

    def tensor_batches(self, metadata, columns, width, entries=None):
        if entries is None:
            chunks = WindowIndex(self, width).chunks()
        else:
            rows = iter(entries)
            chunks = iter(lambda: list(islice(rows, WINDOW_CHUNK_SIZE)), [])
        for chunk in chunks:
            yield self.tensor_batch(chunk, metadata, columns, width)

    def tensor_batch(self, entries, metadata: DatasetMetadata, columns, width):
        layouts = metadata.depth_layouts
        masks = {name: depth_mask_metadata_key(name) for name in layouts.root}
        features = {}
        for column in columns:
            layout = layouts.layout_for_column(column)
            shape = (len(entries), width)
            if layout is not None:
                shape += (layout.max_depth_length,)
            features[column] = torch.zeros(
                shape, dtype=PANDAS_TO_TORCH_TYPES[metadata.column_data_types[column]]
            )
        depth_masks = {
            name: torch.zeros(
                (len(entries), width, layouts.root[name].max_depth_length),
                dtype=torch.bool,
            )
            for name in masks
        }
        selected = [*columns, *masks.values()]
        for sequence, rows in groupby(
            enumerate(entries), key=lambda pair: pair[1]["sequence"]
        ):
            rows = [
                (
                    i,
                    entry,
                    entry["start"] + entry["pad"],
                    min(entry["start"] + width, entry["high"]),
                )
                for i, entry in rows
            ]
            rows = [row for row in rows if row[2] < row[3]]
            if not rows:
                continue
            low, high = min(row[2] for row in rows), max(row[3] for row in rows)
            record = self.dataset["sequences"][str(sequence)]
            last = record["position"] + record["length"]
            low = max(low, record["position"])
            frame = (
                self.read(sequence, low, high, selected)
                if high - low <= width * len(rows) and low < last and selected
                else None
            )
            for i, entry, start, stop in rows:
                if start >= last or not selected:
                    continue
                items = (
                    frame.slice(start - low, stop - start)
                    if frame is not None
                    else self.read(sequence, start, stop, selected)
                )
                output = slice(entry["pad"], entry["pad"] + len(items))
                for column, destination in features.items():
                    values = (
                        items[column].to_list()
                        if destination.ndim == 3
                        else items[column].to_numpy()
                    )
                    destination[i, output] = torch.tensor(
                        np.asarray(values), dtype=destination.dtype
                    )
                for name, column in masks.items():
                    depth_masks[name][i, output] = torch.tensor(
                        items[column].to_list(), dtype=torch.bool
                    )

        def vector(key):
            return torch.tensor([entry[key] for entry in entries], dtype=torch.int64)

        curriculum = entries[0]["curriculum"]
        return StoredTensorBatch(
            sequences=features,
            sequence_ids=vector("sequence"),
            subsequence_ids=vector("id"),
            start_item_positions=vector("start"),
            left_pad_lengths=vector("pad"),
            split_start_item_positions=vector("low"),
            split_end_item_positions=vector("high"),
            depth_valid_masks=depth_masks,
            sample_positions=vector("curriculum") if curriculum else None,
            curriculum_columns=tuple(
                getattr(metadata, "indexed_curriculum_columns", ())
            ),
        )


class WindowIndex:
    """Count and read bounded slices of stored or resized split windows."""

    def __init__(self, view: IndexedView, width: int):
        self.view = view
        self.width = width
        metadata = view._metadata
        split = int(view.view["view"])
        self.stored = width == metadata.width_for_split(split)
        self.stride = getattr(metadata, "window_stride", None) or width
        self.curriculum = getattr(metadata, "indexed_curriculum_columns", ())
        self.alignment = PredictionAlignment.for_split(metadata, split)
        self.ends = []
        self.ranges = view.view["ranges"]
        if self.stored:
            self.count = view.view["window_count"]
        else:
            self.count = 0
            for _, low, high in self.ranges:
                self.count += window_count(
                    high - low, width, self.stride, self.alignment
                )
                self.ends.append(self.count)

    @property
    def chunk_count(self):
        return (self.count + WINDOW_CHUNK_SIZE - 1) // WINDOW_CHUNK_SIZE

    def chunks(self, *, coordinates_only=False):
        if self.stored and self.count:
            columns = ["pad", "start", "low"] if coordinates_only else None
            with pq.ParquetFile(self.view.root / self.view.view["windows"]) as index:
                for group in range(index.num_row_groups):
                    yield index.read_row_group(group, columns=columns).to_pylist()
            return
        for group in range(self.chunk_count):
            yield self._resized_chunk(group, coordinates_only)

    def read_chunk(self, group):
        if self.stored:
            with pq.ParquetFile(self.view.root / self.view.view["windows"]) as index:
                return index.read_row_group(group).to_pylist()
        return self._resized_chunk(group, False)

    def _resized_chunk(self, group, coordinates_only):
        start = group * WINDOW_CHUNK_SIZE
        end = min(start + WINDOW_CHUNK_SIZE, self.count)
        rows = []
        part = bisect_right(self.ends, start)
        while start < end:
            sequence, low, high = self.ranges[part]
            base = self.ends[part - 1] if part else 0
            stop = min(end, self.ends[part])
            placement = WindowPlacement.for_split(
                self.view.dataset["sequences"][str(sequence)]["position"],
                low,
                high,
                self.width,
                self.stride,
                self.alignment,
                begin=start - base,
                end=stop - base,
            )
            for row in _index_rows(sequence, placement, start - base):
                if self.curriculum and not coordinates_only:
                    first = row["start"]
                    selected = self.view.read(
                        sequence,
                        first + row["pad"],
                        first + self.width,
                        self.curriculum,
                    )
                    if any(
                        selected[column].n_unique() != 1 for column in self.curriculum
                    ):
                        raise ValueError(
                            f"Curriculum values vary within window for sequence {sequence}"
                        )
                    row["curriculum"] = selected.row(0)
                rows.append(row)
            start, part = stop, part + 1
        return rows


def feature_contract_id(path):
    """Hash feature semantics independently of split and media locations."""
    metadata = load_dataset_metadata(str(path))
    values = metadata.feature_contract(
        metadata.column_data_types,
        metadata.image_sources,
        use_prefix=metadata.preconditioning_image is not None,
        raw_inputs=True,
    )
    values["static_columns"] = metadata.static_columns
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stored_feature_columns(metadata):
    columns = set(metadata.column_data_types) - set(metadata.static_columns)
    columns.difference_update(metadata.indexed_derived_features)
    columns.update(metadata.indexed_passthrough)
    columns.update(
        depth_mask_metadata_key(name) for name in metadata.depth_layouts.root
    )
    if metadata.indexed_inline_media:
        for name in metadata.image_sources:
            columns.update((f"__image:{name}", f"__image_mask:{name}"))
    return columns | {"sequenceId", "itemPosition"}


class IndexedImageStore:
    """Gather already aligned frames stored in an indexed sequence."""

    def __init__(self, view, sources, prefix):
        self.view, self.sources, self.prefix = view, sources, prefix

    def gather(self, sequence_ids, positions, valid):
        inputs, masks = {}, {}
        for name, source in self.sources.items():
            shape = (*positions.shape, source.frame_capacity)
            inputs[f"__image:{name}"] = torch.zeros(
                (*shape, 3, source.crop_size, source.crop_size), dtype=torch.uint8
            )
            masks[f"__image_mask:{name}"] = torch.zeros(shape, dtype=torch.bool)
        fields = {**inputs, **masks}
        if fields:
            for row, sequence in enumerate(sequence_ids.tolist()):
                steps = valid[row].nonzero().flatten()
                if not len(steps):
                    continue
                items = positions[row, steps]
                first, last = int(items.min()), int(items.max())
                frame = self.view.read(sequence, first, last + 1, list(fields))
                for name, tensor in fields.items():
                    values = torch.as_tensor(frame[name].to_list(), dtype=tensor.dtype)
                    if values.shape != (last - first + 1, *tensor.shape[2:]):
                        raise ValueError(
                            f"{name}: invalid indexed media shape for sequence {sequence}"
                        )
                    tensor[row, steps] = values[items - first]
        if self.prefix is not None:
            ids = sequence_ids.tolist()
            rows = (
                pl.scan_parquet(self.view.static_path)
                .filter(pl.col("sequenceId").is_in(ids))
                .select("sequenceId", PREFIX_IMAGE)
                .collect()
            )
            prefixes = dict(rows.iter_rows())
            inputs[PREFIX_IMAGE] = torch.as_tensor(
                [prefixes[sequence] for sequence in ids], dtype=torch.uint8
            )
        return inputs, masks
