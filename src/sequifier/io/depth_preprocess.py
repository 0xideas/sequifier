"""Ordered, bounded preprocessing for one repeated-row depth layout."""

import hashlib
import heapq
import json
import math
import os
import pickle
import tempfile
from collections import Counter, deque
from contextlib import closing
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Generator

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from loguru import logger

from sequifier.config.depth_layout import DepthLayoutRegistryModel
from sequifier.helpers import PANDAS_TO_TORCH_TYPES
from sequifier.io.pt_payload import (
    StoredTensorBatch,
    concatenate_pt_batches,
    save_pt_payload,
)
from sequifier.io.sample_order import curriculum_columns
from sequifier.special_tokens import SPECIAL_TOKEN_IDS, validate_special_token_ids


@dataclass
class CanonicalItemFrame:
    """A bounded dense window: shallow scalars, depth arrays, and boolean masks."""

    sequences: dict[str, torch.Tensor]
    depth_valid_masks: dict[str, torch.Tensor]


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
            if hasattr(pl.LazyFrame, "collect_batches"):
                batches = (
                    pl.scan_csv(filename)
                    .select(columns)
                    .collect_batches(chunk_size=1024)
                )
                for batch in batches:
                    yield from batch.iter_rows(named=True)
            else:
                reader = pl.read_csv_batched(filename, columns=columns, batch_size=1024)
                while batches := reader.next_batches(1):
                    for batch in batches:
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


def _audit_parquet_groups(path, capacity, n_cores):
    """Count complete items in parallel, joining row groups at item boundaries."""
    from sequifier.preprocess import _create_preprocess_pool

    group_count = pq.ParquetFile(path).num_row_groups
    logger.info(
        "Depth audit: {} Parquet row groups across {} workers",
        group_count,
        min(n_cores, group_count),
    )
    jobs = ((path, index, capacity) for index in range(group_count))
    with _create_preprocess_pool(min(n_cores, group_count)) as pool:
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
            layout.position_base <= child < layout.position_base + layout.context_length
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


def _fit_parquet_group(job):
    from sequifier.preprocess import (
        _apply_configured_input_casting,
        _get_column_statistics,
        _validate_cardinality_reserved_values,
        assign_sequence_to_split,
    )

    (
        path,
        row_groups,
        required,
        columns,
        shallow,
        layout,
        curriculum_columns,
        schema_types,
        configured_types,
        normalize_on_all_data,
        split_method,
        split_ratios,
        seed,
        sequence_counts,
        first_positions,
        assignments,
        precomputed,
        cardinality_config,
        categorical_columns,
        real_columns,
    ) = job
    stats = {}
    categorical_sets = {}
    categorical_counts = {}
    for item in _file_items(
        path,
        "parquet",
        required,
        layout.context_length,
        row_groups=row_groups,
    ):
        sid, pos = item.sequence_id, item.position
        rows, _ = _validated_item_rows(
            item, columns, shallow, layout, curriculum_columns
        )
        for column in cardinality_config:
            observations = (
                [raw[column] for _, raw in rows]
                if column in layout.columns
                else [rows[0][1][column]]
            )
            _validate_cardinality_reserved_values(
                pl.DataFrame(
                    {column: observations}, schema={column: schema_types[column]}
                ),
                {column: cardinality_config[column]},
            )
        fit_on_item = normalize_on_all_data or (
            assignments.get(sid, assign_sequence_to_split(sid, split_ratios, seed)) == 0
            if split_method == "between_sequence"
            else pos
            < first_positions[sid] + int(split_ratios[0] * sequence_counts[sid])
        )
        if not fit_on_item:
            continue
        for column in columns:
            observations = (
                [raw[column] for _, raw in rows]
                if column in layout.columns
                else [rows[0][1][column]]
            )
            data = pl.DataFrame(
                {column: observations}, schema={column: schema_types[column]}
            )
            configured = _observation_types(data, [column], configured_types)
            data = _apply_configured_input_casting(data, [column], configured)
            _, stats = _get_column_statistics(
                data,
                [column],
                {},
                stats,
                0,
                precomputed,
                categorical_value_sets=categorical_sets,
                categorical_columns=categorical_columns,
                real_columns=real_columns,
                categorical_value_counts=categorical_counts,
                cardinality_config=cardinality_config,
            )
    return stats, categorical_sets, categorical_counts


def _render_parquet_sequence(job):
    """Render one complete sequence in a worker with bounded window state."""
    from sequifier.preprocess import (
        _apply_column_statistics,
        _apply_configured_input_casting,
        _apply_output_type_casting,
        get_subsequence_starts,
    )

    path = job["path"]
    sid = job["sequence_id"]
    count = job["count"]
    first_position = job["first_position"]
    layout = job["layout"]
    layouts = job["layouts"]
    name = job["layout_name"]
    columns = job["columns"]
    shallow = job["shallow"]
    curriculum = job["curriculum_columns"]
    schema_types = job["schema_types"]
    col_types = job["col_types"]
    n_classes = job["n_classes"]
    split_paths = job["split_paths"]
    scratch_root = Path(job["scratch_root"])
    completed = job["completed"]
    max_width = max(job["window_lengths"])
    output = {i: [] for i in range(len(job["split_ratios"]))}
    window_counts = Counter()
    expected = set()
    records = {}
    encoded = {}
    order = deque()

    def shard_path(split, index):
        return (
            Path(split_paths[split]).stem
            + f"-0-{job['sequence_index']:012d}{index:012d}.pt"
        )

    def flush(split, index):
        if not output[split]:
            return
        filename = shard_path(split, index)
        destination = scratch_root / filename
        temporary = scratch_root / f"{filename}.tmp"
        batch = concatenate_pt_batches(output[split])
        save_pt_payload(batch, temporary, layouts=layouts, n_classes=n_classes)
        os.replace(temporary, destination)
        checksum = hashlib.sha256()
        with destination.open("rb") as payload:
            for chunk in iter(lambda: payload.read(1024 * 1024), b""):
                checksum.update(chunk)
        records[filename] = {
            "size": destination.stat().st_size,
            "sha256": checksum.hexdigest(),
        }
        output[split].clear()

    stream = iter(
        _file_items(
            path,
            "parquet",
            job["required"],
            layout.context_length,
            row_groups=job["row_groups"],
            sequence_id=sid,
        )
    )
    next_item = next(stream, None)

    def encode_until(stop):
        nonlocal next_item
        while next_item is not None and next_item.position <= stop:
            item = next_item
            rows, item_sample_position = _validated_item_rows(
                item, columns, shallow, layout, curriculum
            )
            data = pl.DataFrame([raw for _, raw in rows], schema=schema_types)
            data = _apply_configured_input_casting(
                data,
                columns,
                _observation_types(data, columns, job["configured_types"]),
            )
            data, _, _ = _apply_column_statistics(
                data,
                columns,
                job["id_maps"],
                job["stats"],
                job["normalize_real_columns"],
                n_classes,
                col_types,
                job["cardinality_config"],
            )
            data = _apply_output_type_casting(data, columns, col_types)
            encoded[item.position] = (rows, data, item_sample_position)
            order.append(item.position)
            if len(order) > max_width:
                encoded.pop(order.popleft())
            next_item = next(stream, None)

    def advance_before(position):
        nonlocal next_item
        while next_item is not None and next_item.position < position:
            next_item = next(stream, None)

    try:
        if job["split_method"] == "within_sequence":
            uppers = [int(x * count) for x in np.cumsum(job["split_ratios"])]
            bounds = [
                (i, low, high)
                for i, (low, high) in enumerate(zip([0] + uppers[:-1], uppers))
            ]
        else:
            bounds = [(job["assigned_split"], 0, count)]
        for split, low, high in bounds:
            width = job["window_lengths"][split]
            if high - low <= 0:
                continue
            alignment = job["alignment"]
            aligned = alignment is not None and split in alignment.splits
            context_low = 0 if aligned else low
            length = high - context_low
            split_start_position = _coordinate(
                first_position + low, "splitStartItemPosition"
            )
            split_last_position = _coordinate(
                first_position + high - 1, "splitEndItemPosition"
            )
            if split_last_position == 2**63 - 1:
                raise ValueError("splitEndItemPosition falls outside signed Int64")
            split_end_position = split_last_position + 1
            pad = max(0, width - length)
            starts = (
                alignment.starts(low, high, width)
                if aligned and alignment is not None
                else get_subsequence_starts(
                    max(length, width), width, job["window_stride"]
                )
            )
            for subsequence, start in enumerate(starts):
                window_pad = max(0, -int(start)) if aligned else pad
                absolute_start = _coordinate(
                    first_position
                    + context_low
                    + int(start)
                    - (0 if aligned else window_pad),
                    "startItemPosition",
                )
                shard_index = window_counts[split] // job["batches_per_file"]
                filename = shard_path(split, shard_index)
                expected.add(filename)
                if filename in completed:
                    advance_before(absolute_start)
                    window_counts[split] += 1
                    continue
                advance_before(absolute_start)
                encode_until(absolute_start + width - 1)
                tensors = {
                    c: torch.zeros(
                        (1, width, layout.context_length)
                        if c in layout.columns
                        else (1, width),
                        dtype=PANDAS_TO_TORCH_TYPES[col_types[c]],
                    )
                    for c in columns
                }
                mask = torch.zeros((1, width, layout.context_length), dtype=torch.bool)
                sample_position = None
                for time in range(window_pad, width):
                    position = absolute_start + time
                    if position not in encoded:
                        raise ValueError(
                            f"Missing depth item {(sid, position)} in ordered stream"
                        )
                    rows, data, item_sample_position = encoded[position]
                    if job["has_sample_positions"]:
                        if sample_position is None:
                            sample_position = item_sample_position
                        elif sample_position != item_sample_position:
                            raise ValueError(
                                "Curriculum columns must be identical within each "
                                "generated subsequence; "
                                f"sequenceId={sid}, subsequenceId={subsequence}"
                            )
                    for row_index, (slot, _) in enumerate(rows):
                        mask[0, time, slot] = True
                        for column in columns:
                            value = data[column][row_index]
                            if column in layout.columns:
                                tensors[column][0, time, slot] = value
                            elif row_index == 0:
                                tensors[column][0, time] = value
                canonical = CanonicalItemFrame(tensors, {name: mask})
                batch = StoredTensorBatch(
                    canonical.sequences,
                    torch.tensor([sid], dtype=torch.int64),
                    torch.tensor([subsequence], dtype=torch.int64),
                    torch.tensor([absolute_start], dtype=torch.int64),
                    torch.tensor([window_pad], dtype=torch.int64),
                    split_start_item_positions=torch.tensor(
                        [split_start_position], dtype=torch.int64
                    ),
                    split_end_item_positions=torch.tensor(
                        [split_end_position], dtype=torch.int64
                    ),
                    depth_valid_masks=canonical.depth_valid_masks,
                    sample_positions=(
                        torch.tensor([sample_position], dtype=torch.int64)
                        if sample_position is not None
                        else None
                    ),
                    curriculum_columns=curriculum,
                )
                batch.validate(layouts, n_classes=n_classes)
                output[split].append(batch)
                window_counts[split] += 1
                if len(output[split]) >= job["batches_per_file"]:
                    flush(split, shard_index)
        for split in output:
            flush(split, window_counts[split] // job["batches_per_file"])
    finally:
        stream.close()
    return expected, records


def preprocess_depth(owner, selected_columns):
    from sequifier.preprocess import (
        _apply_column_statistics,
        _apply_configured_input_casting,
        _apply_output_type_casting,
        _assigned_split_row_counts,
        _balanced_sequence_split_assignments,
        _check_split_window_proportions,
        _estimate_window_proportions,
        _finalize_cardinality_maps,
        _folder_input_files,
        _get_column_statistics,
        _validate_cardinality_columns,
        _validate_cardinality_reserved_values,
        _validate_declared_column_roles,
        _validate_declared_roles_against_metadata,
        assign_sequence_to_split,
        get_subsequence_starts,
        load_precomputed_id_maps,
    )

    if owner.max_rows is not None and owner.max_rows <= 0:
        raise ValueError("Depth max_rows must be a positive logical item count")
    layouts = owner.depth_layouts
    name, layout = next(iter(layouts.items()))
    files = (
        _folder_input_files(owner.preprocessing_data_path, owner.read_format)
        if Path(owner.preprocessing_data_path).is_dir()
        else [owner.preprocessing_data_path]
    )
    configured_curriculum_columns = curriculum_columns(
        getattr(owner, "curriculum_column", None)
    )
    reserved_columns = {
        "sequenceId",
        "itemPosition",
        layout.position_column,
        *configured_curriculum_columns,
    }
    columns = [c for c in selected_columns or [] if c not in reserved_columns]
    _validate_declared_column_roles(
        columns, owner.categorical_columns, owner.real_columns
    )
    scratch_root = Path(owner.project_root) / "data" / owner.target_dir
    schema_types = None
    sample_position_presence = None
    required: list[str] = []
    for filename in files:
        scan = (
            pl.scan_csv(filename)
            if owner.read_format == "csv"
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
                ]
            )
        )
        missing = set(required) - set(schema)
        if missing:
            raise ValueError(f"{filename}: missing columns {sorted(missing)}")
        for coordinate in ("sequenceId", "itemPosition"):
            dtype = schema[coordinate]
            assert dtype.is_integer(), (
                f"{coordinate} must have an integer dtype in {filename}; "
                f"found {dtype}."
            )
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
            _folder_input_files(owner.preprocessing_data_path, owner.read_format)
            if Path(owner.preprocessing_data_path).is_dir()
            else [owner.preprocessing_data_path]
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
        owner.read_format == "parquet"
        and len(files) == 1
        and owner.max_rows is None
        and owner.n_cores > 1
        and pq.ParquetFile(files[0]).num_row_groups > 1
    ):
        (
            sequence_counts,
            first_positions,
            parallel_groups,
            parallel_audits,
        ) = _audit_parquet_groups(files[0], layout.context_length, owner.n_cores)
        selected_count = sum(sequence_counts.values())
        previous = parallel_audits[-1].last
    else:
        for item in _merged_items(
            files,
            owner.read_format,
            ["sequenceId", "itemPosition"],
            layout.context_length,
        ):
            if owner.max_rows is not None and selected_count >= owner.max_rows:
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
            files, owner.read_format, required, layout.context_length, selected_stop
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

    def item_rows(item):
        return _validated_item_rows(
            item, columns, shallow, layout, configured_curriculum_columns
        )

    precomputed = load_precomputed_id_maps(
        owner.project_root, columns, owner.use_precomputed_maps
    )
    _validate_cardinality_columns(owner.cardinality_config, columns, precomputed)
    for column, mapping in precomputed.items():
        if schema_types[column].is_integer():
            precomputed[column] = {
                (int(key) if key not in SPECIAL_TOKEN_IDS.ids_by_label else key): value
                for key, value in mapping.items()
            }
    id_maps, stats = dict(precomputed), {}
    categorical_value_counts = {}
    existing = None
    sequence_counts = dict(sequence_counts)
    assignments = (
        _balanced_sequence_split_assignments(
            list(sequence_counts), owner.split_ratios, owner.seed
        )
        if owner.split_method == "between_sequence"
        else {}
    )
    if owner.split_method == "between_sequence":
        _check_split_window_proportions(
            _estimate_window_proportions(
                _assigned_split_row_counts(
                    sequence_counts,
                    owner.split_ratios,
                    owner.seed,
                    assignments,
                ),
                owner.window_stride,
                owner.alignment,
            )
        )
    first_positions = dict(first_positions)
    if owner.metadata_config_path:
        existing = json.loads(
            (Path(owner.project_root) / owner.metadata_config_path).read_text()
        )
        owner.metadata_fitted_on_all_data = existing.get("normalize_on_all_data", True)
        if (
            DepthLayoutRegistryModel.model_validate(existing.get("depth_layouts", {}))
            != layouts
        ):
            raise ValueError(
                "metadata_config_path requires identical complete depth layouts"
            )
        validate_special_token_ids(
            existing["special_token_ids"], source="depth metadata"
        )
        if existing.get("normalize_real_columns", True) != owner.normalize_real_columns:
            raise ValueError(
                "Metadata normalization policy does not match preprocessing"
            )
        id_maps = existing["id_maps"]
        owner._use_metadata_cardinality_config(existing)
        _validate_cardinality_columns(owner.cardinality_config, columns, {})
        # JSON object keys representing integer categories need their original type.
        for c, mapping in id_maps.items():
            if c in schema_types and schema_types[c].is_integer():
                id_maps[c] = {
                    int(k) if k not in SPECIAL_TOKEN_IDS.ids_by_label else k: v
                    for k, v in mapping.items()
                }
        stats = existing["selected_columns_statistics"]
        _validate_declared_roles_against_metadata(
            owner.categorical_columns,
            owner.real_columns,
            id_maps,
            stats,
            existing.get("column_data_types") or existing.get("column_types"),
        )
    if parallel_groups is not None and existing is None:
        from sequifier.preprocess import (
            _create_preprocess_pool,
            create_id_map,
            get_combined_statistics,
        )

        categorical_sets = {}

        def fit_jobs():
            for group in parallel_groups:
                yield (
                    files[0],
                    group,
                    required,
                    columns,
                    shallow,
                    layout,
                    configured_curriculum_columns,
                    schema_types,
                    owner.column_data_types,
                    owner.normalize_on_all_data,
                    owner.split_method,
                    owner.split_ratios,
                    owner.seed,
                    sequence_counts,
                    first_positions,
                    assignments,
                    precomputed,
                    owner.cardinality_config,
                    owner.categorical_columns,
                    owner.real_columns,
                )

        logger.info(
            "Depth metadata fit: {} source partitions across {} workers",
            len(parallel_groups),
            min(owner.n_cores, len(parallel_groups)),
        )
        with _create_preprocess_pool(min(owner.n_cores, len(parallel_groups))) as pool:
            for local_stats, local_sets, local_counts in pool.imap(
                _fit_parquet_group, fit_jobs()
            ):
                for column, values in local_sets.items():
                    categorical_sets.setdefault(column, set()).update(values)
                for column, counts in local_counts.items():
                    categorical_value_counts.setdefault(column, Counter()).update(
                        counts
                    )
                for column, chunk in local_stats.items():
                    previous_stats = stats.get(column)
                    if previous_stats is None:
                        stats[column] = chunk
                    else:
                        count = int(chunk["count"])
                        previous_count = int(previous_stats["count"])
                        mean, std = get_combined_statistics(
                            count,
                            chunk["mean"],
                            chunk["std"],
                            previous_count,
                            previous_stats["mean"],
                            previous_stats["std"],
                        )
                        stats[column] = {
                            "mean": mean,
                            "std": std,
                            "count": float(count + previous_count),
                        }
        for column, values in categorical_sets.items():
            id_maps[column] = create_id_map(
                pl.DataFrame(
                    {column: list(values)}, schema={column: schema_types[column]}
                ),
                column,
            )
        check_sources()

    for item in (
        [] if parallel_groups is not None and existing is None else selected_items()
    ):
        sid, pos = item.sequence_id, item.position
        rows, _ = item_rows(item)
        for column in owner.cardinality_config:
            observations = (
                [raw[column] for _, raw in rows]
                if column in layout.columns
                else [rows[0][1][column]]
            )
            cardinality_data = pl.DataFrame(
                {column: observations},
                schema={column: schema_types[column]},
            )
            _validate_cardinality_reserved_values(
                cardinality_data, {column: owner.cardinality_config[column]}
            )
        if existing is None:
            fit_on_item = owner.normalize_on_all_data or (
                assignments.get(
                    sid,
                    assign_sequence_to_split(sid, owner.split_ratios, owner.seed),
                )
                == 0
                if owner.split_method == "between_sequence"
                else pos
                < first_positions[sid]
                + int(owner.split_ratios[0] * sequence_counts[sid])
            )
            if not fit_on_item:
                continue
            for column in columns:
                observations = (
                    [raw[column] for _, raw in rows]
                    if column in layout.columns
                    else [rows[0][1][column]]
                )
                data = pl.DataFrame(
                    {column: observations},
                    schema={column: schema_types[column]},
                )
                configured = _observation_types(data, [column], owner.column_data_types)
                data = _apply_configured_input_casting(data, [column], configured)
                id_maps, stats = _get_column_statistics(
                    data,
                    [column],
                    id_maps,
                    stats,
                    0,
                    precomputed,
                    categorical_columns=owner.categorical_columns,
                    real_columns=owner.real_columns,
                    categorical_value_counts=categorical_value_counts,
                    cardinality_config=owner.cardinality_config,
                )
    if existing is None:
        id_maps = _finalize_cardinality_maps(
            id_maps,
            categorical_value_counts,
            owner.cardinality_config,
        )
    col_types = (
        owner.column_data_types
        or (existing or {}).get("column_data_types")
        or {
            c: (
                "Int64"
                if c in id_maps
                else (
                    "Float64"
                    if c in stats and not schema_types[c].is_float()
                    else str(schema_types[c])
                )
            )
            for c in columns
        }
    )
    if set(columns) - set(col_types):
        raise ValueError("Column dtypes must cover selected features")
    if existing and any(
        col_types[c] != existing["column_data_types"].get(c) for c in columns
    ):
        raise ValueError("Configured output types differ from reused metadata")
    n_classes = {
        c: max(SPECIAL_TOKEN_IDS.user_start, max(mapping.values()) + 1)
        for c, mapping in id_maps.items()
        if c in columns
    }
    owner._write_or_validate_resume_manifest(
        selected_columns, "pt", columns, id_maps, n_classes, col_types, stats
    )
    owner._export_metadata(id_maps, n_classes, col_types, stats)
    # A shard is reusable only after both its payload and its ledger entry have
    # been committed. The ledger also ties completed shards to this source set.
    ledger_path = scratch_root / "depth-shards.json"
    source_identity = [list(entry) for entry in source_snapshot]
    parallel_render = parallel_groups is not None
    if owner.continue_preprocessing and ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if (
            ledger.get("version") not in (1, 2)
            or ledger.get("sources") != source_identity
        ):
            raise ValueError(
                "Depth resume source files differ from the completed shards"
            )
        parallel_render = parallel_render and ledger["version"] == 2
        if ledger["version"] == 2 and not parallel_render:
            raise ValueError(
                "Depth parallel resume requires the original Parquet row groups "
                "and n_cores > 1"
            )
    else:
        ledger = {
            "version": 2 if parallel_render else 1,
            "sources": source_identity,
            "shards": {},
        }

    def shard_path(split, index):
        return Path(owner.split_paths[split]).stem + f"-0-{index:08d}.pt"

    def shard_location(split, filename):
        return Path(owner.split_paths[split]).with_suffix("") / filename

    def digest(path):
        checksum = hashlib.sha256()
        with path.open("rb") as payload:
            for chunk in iter(lambda: payload.read(1024 * 1024), b""):
                checksum.update(chunk)
        return checksum.hexdigest()

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
                and digest(scratch) == record["sha256"]
            )
            valid_final = (
                final.is_file()
                and final.stat().st_size == record["size"]
                and digest(final) == record["sha256"]
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
            for index, (sid, count) in enumerate(sequence_counts.items()):
                yield {
                    "path": files[0],
                    "row_groups": [
                        audit.index for audit in row_group_audits if sid in audit.counts
                    ],
                    "required": required,
                    "sequence_id": sid,
                    "sequence_index": index,
                    "count": count,
                    "first_position": first_positions[sid],
                    "layout": layout,
                    "layouts": layouts,
                    "layout_name": name,
                    "columns": columns,
                    "shallow": shallow,
                    "curriculum_columns": configured_curriculum_columns,
                    "schema_types": schema_types,
                    "configured_types": owner.column_data_types,
                    "col_types": col_types,
                    "n_classes": n_classes,
                    "id_maps": id_maps,
                    "stats": stats,
                    "normalize_real_columns": owner.normalize_real_columns,
                    "cardinality_config": owner.cardinality_config,
                    "split_paths": owner.split_paths,
                    "scratch_root": str(scratch_root),
                    "completed": completed,
                    "window_lengths": owner.window_lengths,
                    "split_ratios": owner.split_ratios,
                    "split_method": owner.split_method,
                    "assigned_split": assignments.get(
                        sid,
                        assign_sequence_to_split(sid, owner.split_ratios, owner.seed),
                    ),
                    "alignment": owner.alignment,
                    "window_stride": owner.window_stride,
                    "batches_per_file": owner.batches_per_file,
                    "has_sample_positions": owner.has_sample_positions,
                }

        logger.info(
            "Depth windows: {} sequences across {} workers",
            len(sequence_counts),
            min(owner.n_cores, len(sequence_counts)),
        )
        with _create_preprocess_pool(min(owner.n_cores, len(sequence_counts))) as pool:
            for names, records in pool.imap_unordered(
                _render_parquet_sequence, render_jobs()
            ):
                expected_shards.update(names)
                ledger["shards"].update(records)
                commit_ledger()
        check_sources()
        for stale in scratch_root.glob("*.pt"):
            if stale.name not in expected_shards:
                raise ValueError(
                    f"Depth temp folder contains stale output {stale}; "
                    "use a fresh output location"
                )
        for split_path in owner.split_paths:
            folder = Path(split_path).with_suffix("")
            if folder.exists():
                for stale in folder.glob("*.pt"):
                    if stale.name not in expected_shards:
                        raise ValueError(
                            f"Existing split folder contains stale output {stale}; "
                            "use a fresh output location"
                        )
        owner._cleanup("pt")
        return

    # Materialize only requested windows, never a whole sequence/folder densely.
    max_width = max(owner.window_lengths)
    output = {i: [] for i in range(len(owner.split_ratios))}
    window_counts = Counter()

    def flush(split, index):
        if not output[split]:
            return
        filename = shard_path(split, index)
        destination = scratch_root / filename
        temporary = scratch_root / f"{filename}.tmp"
        batch = concatenate_pt_batches(output[split])
        save_pt_payload(batch, temporary, layouts=layouts, n_classes=n_classes)
        os.replace(temporary, destination)
        ledger["shards"][filename] = {
            "size": destination.stat().st_size,
            "sha256": digest(destination),
        }
        commit_ledger()
        output[split].clear()

    # Encode each item once and retain only the previous window's worth of items.
    encoded = {}
    order = deque()
    stream = iter(selected_items())
    next_item = next(stream, None)

    def encode_until(sid, stop):
        nonlocal next_item
        while (
            next_item is not None
            and next_item.sequence_id == sid
            and next_item.position <= stop
        ):
            item = next_item
            rows, item_sample_position = item_rows(item)
            data = pl.DataFrame([raw for _, raw in rows], schema=schema_types)
            data = _apply_configured_input_casting(
                data,
                columns,
                _observation_types(data, columns, owner.column_data_types),
            )
            data, _, _ = _apply_column_statistics(
                data,
                columns,
                id_maps,
                stats,
                owner.normalize_real_columns,
                n_classes,
                col_types,
                owner.cardinality_config,
            )
            data = _apply_output_type_casting(data, columns, col_types)
            encoded[item.position] = (rows, data, item_sample_position)
            order.append(item.position)
            if len(order) > max_width:
                encoded.pop(order.popleft())
            next_item = next(stream, None)

    def advance_before(sid, position):
        """Discard items preceding the next window without encoding them."""
        nonlocal next_item
        while next_item is not None and (
            next_item.sequence_id < sid
            or (next_item.sequence_id == sid and next_item.position < position)
        ):
            next_item = next(stream, None)

    try:
        for sid, count in sequence_counts.items():
            if owner.split_method == "within_sequence":
                uppers = [int(x * count) for x in np.cumsum(owner.split_ratios)]
                bounds = [
                    (i, low, high)
                    for i, (low, high) in enumerate(zip([0] + uppers[:-1], uppers))
                ]
            else:
                assigned_split = assignments.get(
                    sid, assign_sequence_to_split(sid, owner.split_ratios, owner.seed)
                )
                bounds = [(assigned_split, 0, count)]
            first_position = first_positions[sid]
            encoded.clear()
            order.clear()
            for split, low, high in bounds:
                width = owner.window_lengths[split]
                target_length = high - low
                if target_length <= 0:
                    continue
                aligned = (
                    owner.alignment is not None and split in owner.alignment.splits
                )
                context_low = 0 if aligned else low
                length = high - context_low
                split_start_position = _coordinate(
                    first_position + low, "splitStartItemPosition"
                )
                split_last_position = _coordinate(
                    first_position + high - 1, "splitEndItemPosition"
                )
                if split_last_position == 2**63 - 1:
                    raise ValueError("splitEndItemPosition falls outside signed Int64")
                split_end_position = split_last_position + 1
                pad = max(0, width - length)
                starts = (
                    owner.alignment.starts(low, high, width)
                    if aligned and owner.alignment is not None
                    else get_subsequence_starts(
                        max(length, width), width, owner.window_stride
                    )
                )
                for subsequence, start in enumerate(starts):
                    window_pad = max(0, -int(start)) if aligned else pad
                    absolute_start = _coordinate(
                        first_position
                        + context_low
                        + int(start)
                        - (0 if aligned else window_pad),
                        "startItemPosition",
                    )
                    shard_index = window_counts[split] // owner.batches_per_file
                    if shard_path(split, shard_index) in completed:
                        advance_before(sid, absolute_start)
                        window_counts[split] += 1
                        continue
                    advance_before(sid, absolute_start)
                    encode_until(sid, absolute_start + width - 1)
                    tensors = {
                        c: torch.zeros(
                            (1, width, layout.context_length)
                            if c in layout.columns
                            else (1, width),
                            dtype=PANDAS_TO_TORCH_TYPES[col_types[c]],
                        )
                        for c in columns
                    }
                    mask = torch.zeros(
                        (1, width, layout.context_length), dtype=torch.bool
                    )
                    sample_position = None
                    for time in range(window_pad, width):
                        position = absolute_start + time
                        if position not in encoded:
                            raise ValueError(
                                f"Missing depth item {(sid, position)} "
                                "in ordered stream"
                            )
                        rows, data, item_sample_position = encoded[position]
                        if owner.has_sample_positions:
                            if sample_position is None:
                                sample_position = item_sample_position
                            elif sample_position != item_sample_position:
                                raise ValueError(
                                    "Curriculum columns must be identical within each "
                                    "generated subsequence; "
                                    f"sequenceId={sid}, subsequenceId={subsequence}"
                                )
                        for row_index, (slot, _) in enumerate(rows):
                            mask[0, time, slot] = True
                            for column in columns:
                                value = data[column][row_index]
                                if column in layout.columns:
                                    tensors[column][0, time, slot] = value
                                elif row_index == 0:
                                    tensors[column][0, time] = value
                    canonical = CanonicalItemFrame(tensors, {name: mask})
                    batch = StoredTensorBatch(
                        canonical.sequences,
                        torch.tensor([sid], dtype=torch.int64),
                        torch.tensor([subsequence], dtype=torch.int64),
                        torch.tensor([absolute_start], dtype=torch.int64),
                        torch.tensor([window_pad], dtype=torch.int64),
                        split_start_item_positions=torch.tensor(
                            [split_start_position], dtype=torch.int64
                        ),
                        split_end_item_positions=torch.tensor(
                            [split_end_position], dtype=torch.int64
                        ),
                        depth_valid_masks=canonical.depth_valid_masks,
                        sample_positions=(
                            torch.tensor([sample_position], dtype=torch.int64)
                            if sample_position is not None
                            else None
                        ),
                        curriculum_columns=configured_curriculum_columns,
                    )
                    batch.validate(layouts, n_classes=n_classes)
                    output[split].append(batch)
                    window_counts[split] += 1
                    if len(output[split]) >= owner.batches_per_file:
                        flush(split, shard_index)
        for split in output:
            flush(split, window_counts[split] // owner.batches_per_file)
    finally:
        stream.close()
    expected_shards = {
        shard_path(split, index)
        for split in range(len(owner.split_paths))
        for index in range(
            (window_counts[split] + owner.batches_per_file - 1)
            // owner.batches_per_file
        )
    }
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
                    raise ValueError(
                        f"Existing split folder contains stale output {stale}; "
                        "use a fresh output location"
                    )
    owner._cleanup("pt")
