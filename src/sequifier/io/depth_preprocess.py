"""Disk-backed complete-item preprocessing for one repeated-row depth layout.

Raw fragments are indexed before validation. Dense memory is bounded by one
stored window plus the configured output batch, independently of folder size.
"""

import json
import math
import pickle
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch

from sequifier.config.depth_layout import DepthLayoutRegistryModel
from sequifier.helpers import PANDAS_TO_TORCH_TYPES
from sequifier.io.pt_payload import (
    StoredTensorBatch,
    concatenate_pt_batches,
    save_pt_payload,
)
from sequifier.special_tokens import SPECIAL_TOKEN_IDS, validate_special_token_ids


@dataclass
class CanonicalItemFrame:
    """A bounded dense window: shallow scalars, depth arrays, and boolean masks."""

    sequences: dict[str, torch.Tensor]
    depth_valid_masks: dict[str, torch.Tensor]


def _coordinate(value, name, *, integer_only=False):
    if (
        value is None
        or isinstance(value, bool)
        or (integer_only and not isinstance(value, int))
    ):
        raise ValueError(f"{name} must be a non-null integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be representable as signed Int64") from error
    if isinstance(value, float) and value != result or not -(2**63) <= result < 2**63:
        raise ValueError(f"{name} must fit signed Int64 without loss")
    return result


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


def preprocess_depth(owner, selected_columns):
    from sequifier.preprocess import (
        _apply_column_statistics,
        _apply_configured_input_casting,
        _apply_output_type_casting,
        _balanced_sequence_split_assignments,
        _folder_input_files,
        _get_column_statistics,
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
    columns = [
        c for c in selected_columns or [] if c not in {"sequenceId", "itemPosition"}
    ]
    scratch_root = Path(owner.project_root) / "data" / owner.target_dir
    with tempfile.TemporaryDirectory(prefix="depth-index-", dir=scratch_root) as tmp:
        db = sqlite3.connect(str(Path(tmp) / "items.sqlite"))
        try:
            db.execute("PRAGMA temp_store=FILE")
            db.execute(
                "CREATE TABLE raw (sid INTEGER, pos INTEGER, child BLOB, values_blob BLOB)"
            )
            schema_types = None
            for file_index, filename in enumerate(files):
                scan = (
                    pl.scan_csv(filename)
                    if owner.read_format == "csv"
                    else pl.scan_parquet(filename)
                )
                schema = scan.collect_schema()
                if layout.position_column not in schema:
                    raise ValueError(
                        f"{filename} is missing depth position column {layout.position_column!r}"
                    )
                if not columns:
                    columns = [
                        c
                        for c in schema
                        if c
                        not in {"sequenceId", "itemPosition", layout.position_column}
                    ]
                if not set(layout.columns) <= set(columns):
                    raise ValueError(
                        "Selected features must include every depth layout column"
                    )
                required = [
                    "sequenceId",
                    "itemPosition",
                    layout.position_column,
                    *columns,
                ]
                if set(required) - set(schema):
                    raise ValueError(
                        f"{filename}: missing columns {sorted(set(required) - set(schema))}"
                    )
                types = {c: schema[c] for c in columns}
                if schema_types is not None and types != schema_types:
                    raise ValueError(
                        "Depth source files must use consistent selected feature dtypes"
                    )
                schema_types = types
                spool = Path(tmp) / f"source-{file_index}.parquet"
                scan.select(required).sink_parquet(spool)
                for batch in pq.ParquetFile(spool).iter_batches(batch_size=16384):
                    rows = []
                    for row in batch.to_pylist():
                        sid = _coordinate(row["sequenceId"], "sequenceId")
                        pos = _coordinate(
                            row["itemPosition"], "itemPosition", integer_only=True
                        )
                        rows.append(
                            (
                                sid,
                                pos,
                                pickle.dumps(row[layout.position_column]),
                                pickle.dumps({c: row[c] for c in columns}),
                            )
                        )
                    db.executemany("INSERT INTO raw VALUES (?, ?, ?, ?)", rows)
                db.commit()
                spool.unlink()
            if not schema_types:
                raise ValueError("Depth preprocessing source is empty")
            db.execute("CREATE INDEX raw_coordinates ON raw (sid, pos)")
            db.execute(
                "CREATE TABLE items AS SELECT DISTINCT sid, pos FROM raw ORDER BY sid, pos LIMIT ?",
                (owner.max_rows if owner.max_rows is not None else -1,),
            )
            db.execute("CREATE UNIQUE INDEX item_coordinates ON items (sid, pos)")
            if db.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0:
                raise ValueError("No logical items selected for depth preprocessing")
            # A selected prefix can stop inside a sequence, but never inside an item.
            previous = None
            for sid, pos in db.execute("SELECT sid, pos FROM items ORDER BY sid, pos"):
                if (
                    previous is not None
                    and sid == previous[0]
                    and pos != previous[1] + 1
                ):
                    raise ValueError(
                        f"itemPosition must be continuous within sequence {sid}"
                    )
                previous = (sid, pos)
            shallow = [c for c in columns if c not in layout.columns]

            def item_rows(sid, pos):
                records = db.execute(
                    "SELECT child, values_blob FROM raw WHERE sid=? AND pos=?",
                    (sid, pos),
                ).fetchmany(layout.context_length + 1)
                if len(records) > layout.context_length:
                    raise ValueError(f"Item {(sid, pos)} exceeds depth capacity")
                values = []
                seen = set()
                for child_blob, raw_blob in records:
                    child = _coordinate(
                        pickle.loads(child_blob),
                        layout.position_column,
                        integer_only=True,
                    )
                    if (
                        not layout.position_base
                        <= child
                        < layout.position_base + layout.context_length
                    ):
                        raise ValueError(
                            f"Depth position {child} is outside layout capacity"
                        )
                    slot = child - layout.position_base
                    if slot in seen:
                        raise ValueError(
                            f"Duplicate outer/depth coordinate {(sid, pos, child)}"
                        )
                    seen.add(slot)
                    raw = pickle.loads(raw_blob)
                    for column, value in raw.items():
                        if (
                            value is None
                            or isinstance(value, float)
                            and not math.isfinite(value)
                        ):
                            raise ValueError(
                                f"{column}: selected observations must be non-null and finite"
                            )
                    if values and any(raw[c] != values[0][1][c] for c in shallow):
                        raise ValueError(
                            f"Shallow values disagree for outer item {(sid, pos)}"
                        )
                    values.append((slot, raw))
                if not layout.allow_gaps and sorted(seen) != list(range(len(seen))):
                    raise ValueError(f"Item {(sid, pos)} has forbidden depth gaps")
                return sorted(values)

            precomputed = load_precomputed_id_maps(
                owner.project_root, columns, owner.use_precomputed_maps
            )
            for column, mapping in precomputed.items():
                if schema_types[column].is_integer():
                    precomputed[column] = {
                        int(key)
                        if key not in SPECIAL_TOKEN_IDS.ids_by_label
                        else key: value
                        for key, value in mapping.items()
                    }
            id_maps, stats = dict(precomputed), {}
            existing = None
            if owner.metadata_config_path:
                existing = json.loads(
                    (Path(owner.project_root) / owner.metadata_config_path).read_text()
                )
                if (
                    DepthLayoutRegistryModel.model_validate(
                        existing.get("depth_layouts", {})
                    )
                    != layouts
                ):
                    raise ValueError(
                        "metadata_config_path requires identical complete depth layouts"
                    )
                validate_special_token_ids(
                    existing["special_token_ids"], source="depth metadata"
                )
                if (
                    existing.get("normalize_real_columns", True)
                    != owner.normalize_real_columns
                ):
                    raise ValueError(
                        "Metadata normalization policy does not match preprocessing"
                    )
                id_maps = existing["id_maps"]
                # JSON object keys representing integer categories need their original type.
                for c, mapping in id_maps.items():
                    if c in schema_types and schema_types[c].is_integer():
                        id_maps[c] = {
                            int(k) if k not in SPECIAL_TOKEN_IDS.ids_by_label else k: v
                            for k, v in mapping.items()
                        }
                stats = existing["selected_columns_statistics"]
            for sid, pos in db.execute("SELECT sid, pos FROM items ORDER BY sid, pos"):
                rows = item_rows(sid, pos)
                if existing is None:
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
                        configured = _observation_types(
                            data, [column], owner.column_data_types
                        )
                        data = _apply_configured_input_casting(
                            data, [column], configured
                        )
                        id_maps, stats = _get_column_statistics(
                            data, [column], id_maps, stats, 0, precomputed
                        )
            col_types = (
                owner.column_data_types
                or (existing or {}).get("column_data_types")
                or {
                    c: "Int64" if c in id_maps else str(schema_types[c])
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
                c: max(mapping.values()) + 1
                for c, mapping in id_maps.items()
                if c in columns
            }
            owner._write_or_validate_resume_manifest(
                selected_columns, "pt", columns, id_maps, n_classes, col_types, stats
            )
            owner._export_metadata(id_maps, n_classes, col_types, stats)
            # Materialize only requested windows, never a whole sequence/folder densely.
            sequence_counts = dict(
                db.execute("SELECT sid, COUNT(*) FROM items GROUP BY sid ORDER BY sid")
            )
            assignments = (
                _balanced_sequence_split_assignments(
                    list(sequence_counts), owner.split_ratios, owner.seed
                )
                if owner.split_method == "between_sequence"
                else {}
            )
            width = owner.storage_layout.window_length
            output = {i: [] for i in range(len(owner.split_ratios))}
            file_numbers = Counter()

            def flush(split):
                if not output[split]:
                    return
                filename = (
                    Path(owner.split_paths[split]).stem
                    + f"-0-{file_numbers[split]:08d}.pt"
                )
                destination = scratch_root / filename
                batch = concatenate_pt_batches(output[split])
                save_pt_payload(
                    batch, destination, layouts=layouts, n_classes=n_classes
                )
                output[split].clear()
                file_numbers[split] += 1

            for sid, count in sequence_counts.items():
                if owner.split_method == "within_sequence":
                    uppers = [int(x * count) for x in np.cumsum(owner.split_ratios)]
                    bounds = [
                        (i, low, high)
                        for i, (low, high) in enumerate(zip([0] + uppers[:-1], uppers))
                    ]
                else:
                    bounds = [(assignments[sid], 0, count)]
                first_position = db.execute(
                    "SELECT MIN(pos) FROM items WHERE sid=?", (sid,)
                ).fetchone()[0]
                for split, low, high in bounds:
                    length = high - low
                    if length <= 0:
                        continue
                    pad = max(0, width - length)
                    starts = get_subsequence_starts(
                        max(length, width),
                        width,
                        owner.window_strides[split],
                        owner.window_placement,
                    )
                    for subsequence, start in enumerate(starts):
                        absolute_start = _coordinate(
                            first_position + low + int(start) - pad,
                            "startItemPosition",
                        )
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
                        for time in range(pad, width):
                            position = absolute_start + time
                            rows = item_rows(sid, position)
                            data = pl.DataFrame(
                                [raw for _, raw in rows], schema=schema_types
                            )
                            data = _apply_configured_input_casting(
                                data,
                                columns,
                                _observation_types(
                                    data, columns, owner.column_data_types
                                ),
                            )
                            data, _, _ = _apply_column_statistics(
                                data,
                                columns,
                                id_maps,
                                stats,
                                owner.normalize_real_columns,
                                n_classes,
                                col_types,
                            )
                            data = _apply_output_type_casting(data, columns, col_types)
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
                            torch.tensor([pad], dtype=torch.int64),
                            canonical.depth_valid_masks,
                        )
                        batch.validate(layouts, n_classes=n_classes)
                        output[split].append(batch)
                        if len(output[split]) >= owner.batches_per_file:
                            flush(split)
            for split in output:
                flush(split)
        finally:
            db.close()
    for split_path in owner.split_paths:
        folder = Path(split_path).with_suffix("")
        if folder.exists():
            for stale in folder.glob("*.pt"):
                if stale.name not in {path.name for path in scratch_root.glob("*.pt")}:
                    raise ValueError(
                        f"Existing split folder contains stale output {stale}; use a fresh output location"
                    )
    owner._cleanup("pt")
