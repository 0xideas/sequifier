"""Persistent item storage and coordinate-only views of sequence windows."""

import hashlib
import json
import os
from bisect import bisect_right
from dataclasses import dataclass
from itertools import groupby, islice
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from sequifier.config.depth_layout import depth_mask_metadata_key
from sequifier.config.metadata import DatasetMetadata, load_dataset_metadata
from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.helpers import PANDAS_TO_TORCH_TYPES, StoredWindowLayout
from sequifier.io.pt_payload import StoredTensorBatch
from sequifier.trajectory.sources import file_checksum

DATASET_FORMAT = "sequifier.indexed-sequences"
VIEW_FORMAT = "sequifier.sequence-view"
ROW_GROUP_SIZE = 8192
WINDOW_CHUNK_SIZE = 256
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


class SplitIndexWriter:
    def __init__(self, path):
        self.path = path
        self.ranges = {}
        self.count = 0
        self.buffer = []
        self.writer = pq.ParquetWriter(path, WINDOW_INDEX_SCHEMA)

    def add(self, sequence_id, view):
        self.ranges[sequence_id] = [
            int(view.split_start_item_positions[0]),
            int(view.split_end_item_positions[0]),
        ]
        for i in range(len(view.subsequence_ids)):
            self.buffer.append(
                {
                    "sequence": sequence_id,
                    "id": int(view.subsequence_ids[i]),
                    "start": int(view.start_item_positions[i]),
                    "pad": int(view.left_pad_lengths[i]),
                    "low": int(view.split_start_item_positions[i]),
                    "high": int(view.split_end_item_positions[i]),
                    "curriculum": (
                        view.sample_positions[i].tolist()
                        if view.sample_positions is not None
                        else []
                    ),
                }
            )
            self.count += 1
            if len(self.buffer) == ROW_GROUP_SIZE:
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

    def __init__(self, output_path):
        self.destination = Path(output_path).absolute()
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        if self.destination.exists():
            raise FileExistsError(self.destination)
        self.root = Path(
            mkdtemp(prefix=f".{self.destination.name}-", dir=self.destination.parent)
        )
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

    def add(self, items, windows, *, episode_id=None, group_id=None, content_key=None):
        sequence_ids = items["sequenceId"].unique()
        if len(sequence_ids) != 1 or items.is_empty():
            raise ValueError("Indexed sequence writes require one nonempty sequence")
        sequence_id = int(sequence_ids.item())
        if sequence_id in self.sequences:
            raise ValueError(f"Duplicate indexed sequence {sequence_id}")
        path = self.root / f"items-{len(self.sequences):08d}.parquet"
        items.write_parquet(path, row_group_size=ROW_GROUP_SIZE)
        self.files.append(path)
        positions = items["itemPosition"]
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
        for split, view in windows.items():
            if view is None:
                continue
            name = str(split)
            if name not in self.views:
                self.views[name] = SplitIndexWriter(
                    self.root / f"{name}-windows.parquet"
                )
            self.views[name].add(sequence_id, view)

    def split_sequence_ids(self, split):
        view = self.views.get(str(split))
        return set(view.ranges) if view else set()

    def write_static(self, static):
        path = self.root / "static.parquet"
        static.write_parquet(path)
        self.files.append(path)

    def publish(self, metadata, *, original_metadata=None):
        version = ContractVersions.INDEXED_SEQUENCES
        metadata_path = self.root / "metadata.json"
        write_json(metadata_path, metadata)
        self.files.append(metadata_path)
        if original_metadata is None:
            original_path = metadata_path
        else:
            original_path = self.root / "original-metadata.json"
            if Path(original_metadata).resolve() != original_path.resolve():
                original_path.write_bytes(Path(original_metadata).read_bytes())
            self.files.append(original_path)
        for name in metadata["indexed_views"]:
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
                    "ranges": index.ranges if index else {},
                },
            )
        self.files.extend(self.root.glob("*.split.json"))
        write_json(
            self.root / "dataset.json",
            {
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
            },
        )
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

    def verify(self):
        dependencies = {
            self.path: file_checksum(self.path),
            self.manifest_path: file_checksum(self.manifest_path),
        }
        for name, expected in self.dataset["checksums"].items():
            path = self.root / name
            if file_checksum(path) != expected:
                raise ValueError(f"Indexed dataset checksum disagrees with {path}")
            dependencies[path] = expected
        return dependencies

    def metadata(self):
        return load_dataset_metadata(str(self.metadata_path))

    def split_bounds(self):
        return {
            entry["sequence"]: (entry["low"], entry["high"]) for entry in self.entries()
        }

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
        if index in metadata.prediction_aligned_splits:
            width += (metadata.prediction_alignment_steps - 1) * (
                metadata.prediction_length or 1
            )
        return StoredWindowLayout(max(width, base.window_length), offset, base.version)

    def entries(self, width=None):
        windows = self.view["windows"]
        rows = pl.read_parquet(self.root / windows).to_dicts() if windows else []
        if width is None:
            return rows
        metadata = self.metadata()
        index = int(self.view["view"])
        if width == metadata.width_for_split(index):
            return rows
        return list(self._place_windows(rows, index, metadata, width))

    def _place_windows(self, rows, index, metadata, width):
        from sequifier.preprocess import PredictionAlignment, get_subsequence_starts

        stride = getattr(metadata, "window_stride", None) or width
        curriculum = getattr(metadata, "indexed_curriculum_columns", [])
        aligned = index in metadata.prediction_aligned_splits
        for sequence, entries in groupby(rows, key=lambda row: row["sequence"]):
            first = next(entries)
            low, high = first["low"], first["high"]
            start = (
                self.dataset["sequences"][str(sequence)]["position"] if aligned else low
            )
            if aligned:
                alignment = PredictionAlignment(
                    (index,),
                    metadata.prediction_length,
                    metadata.target_offset,
                    metadata.prediction_alignment_steps,
                )
                offsets = alignment.starts(low - start, high - start, width)
            else:
                span = high - start
                pad = max(0, width - span)
                offsets = get_subsequence_starts(span + pad, width, stride) - pad
            for subsequence, offset in enumerate(offsets):
                position = start + int(offset)
                pad = max(0, start - position)
                values = []
                if curriculum:
                    selected = self.read(
                        sequence, position + pad, position + width, curriculum
                    )
                    if any(selected[column].n_unique() != 1 for column in curriculum):
                        raise ValueError(
                            f"Curriculum values vary within window for sequence {sequence}"
                        )
                    values = selected.row(0)
                yield {
                    "sequence": sequence,
                    "id": subsequence,
                    "start": position,
                    "pad": pad,
                    "low": low,
                    "high": high,
                    "curriculum": values,
                }

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

    def read(self, sequence, start=None, end=None, columns=None):
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
            return pl.from_arrow(table.slice(low - offsets[first_group], high - low))

    def tensor_batches(self, metadata, columns, width, entries=None):
        if entries is None and self.view["windows"] is not None:
            base = self.metadata().width_for_split(int(self.view["view"]))
            if width == base:
                with pq.ParquetFile(self.root / self.view["windows"]) as index:
                    for chunk in index.iter_batches(batch_size=WINDOW_CHUNK_SIZE):
                        yield self.tensor_batch(
                            chunk.to_pylist(), metadata, columns, width
                        )
                return
        rows = iter(self.entries(width) if entries is None else entries)
        while chunk := list(islice(rows, WINDOW_CHUNK_SIZE)):
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


def feature_contract_id(path):
    """Hash feature semantics independently of split and media locations."""
    metadata = load_dataset_metadata(str(path))
    values = metadata.model_dump(
        mode="json",
        include={
            "column_data_types",
            "static_columns",
            "depth_layouts",
            "id_maps",
            "n_classes",
            "cardinality_config",
            "special_token_ids",
            "selected_columns_statistics",
            "normalize_real_columns",
            "time_alignment_column",
        },
    )
    values["raw_mask_column"] = getattr(metadata, "raw_mask_column", None)
    values["image_sources"] = {
        name: source.model_dump(
            mode="json",
            exclude={
                "path",
                "signature",
                "time_index_path",
                "time_index_signature",
                "video_index_paths",
            },
        )
        for name, source in metadata.image_sources.items()
    }
    prefix = metadata.preconditioning_image
    if prefix is not None:
        values["preconditioning_image"] = prefix.model_dump(
            mode="json", exclude={"paths", "signature"}
        )
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
