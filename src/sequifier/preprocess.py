import hashlib
import heapq
import json
import math
import multiprocessing
import os
import re
import shutil
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from loguru import logger

from sequifier.config.depth_layout import DepthLayoutRegistryModel
from sequifier.config.preprocess_config import (
    CardinalityLimitModel,
    load_preprocessor_config,
)
from sequifier.helpers import (
    PANDAS_TO_TORCH_TYPES,
    StoredWindowLayout,
    assign_sequence_to_split,
    canonicalize_polars_dtype_name,
    is_float_dtype_name,
    is_integer_dtype_name,
    polars_dtype_from_name,
    read_data,
)
from sequifier.io.pt_payload import StoredTensorBatch, load_pt_payload, save_pt_payload
from sequifier.io.sample_order import (
    CURRICULUM_COLUMN_PREFIX,
    curriculum_columns,
    curriculum_storage_column,
    curriculum_storage_columns,
    source_curriculum_column,
)
from sequifier.special_tokens import (
    SPECIAL_TOKEN_ID_VALUES,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKEN_LABELS,
    validate_special_token_ids,
)
from sequifier.typechecking import beartype

INPUT_METADATA_COLUMNS = ("sequenceId", "itemPosition")
SPLIT_VALUE_COLUMN = "__sequifier_split_value"
REAL_MASK_VALUE = 0.0
CURRENT_STORED_WINDOW_LAYOUT_VERSION = 2
MAX_WINDOW_BUFFER_BYTES = 256 * 1024 * 1024
HASH_BUCKET_LABEL_PREFIX = "[hash_bucket:"
STABLE_HASH_OFFSET_BASIS = 14695981039346656037
STABLE_HASH_PRIME = 1099511628211
UINT64_MASK = (1 << 64) - 1

FLOAT_TYPE_ORDER = ("Float16", "Float32", "Float64")
INTEGER_TYPE_ORDER = (
    "Int8",
    "UInt8",
    "Int16",
    "UInt16",
    "Int32",
    "UInt32",
    "Int64",
    "UInt64",
)
INTEGER_TYPE_INFO = {
    "Int8": np.iinfo(np.int8),
    "Int16": np.iinfo(np.int16),
    "Int32": np.iinfo(np.int32),
    "Int64": np.iinfo(np.int64),
    "UInt8": np.iinfo(np.uint8),
    "UInt16": np.iinfo(np.uint16),
    "UInt32": np.iinfo(np.uint32),
    "UInt64": np.iinfo(np.uint64),
}
FLOAT_TYPE_INFO = {
    "Float16": np.finfo(np.float16),
    "Float32": np.finfo(np.float32),
    "Float64": np.finfo(np.float64),
}
FLOAT_EXACT_INTEGER_LIMITS = {
    "Float16": 2**11,
    "Float32": 2**24,
    "Float64": 2**53,
}
INT64_INFO = np.iinfo(np.int64)


@dataclass(frozen=True)
class SequenceWindows:
    """Dense windows and metadata extracted from one sequence split."""

    sequence_id: int
    subsequence_ids: np.ndarray
    start_item_positions: np.ndarray
    left_pad_lengths: np.ndarray
    split_start_item_positions: np.ndarray
    split_end_item_positions: np.ndarray
    values: dict[str, np.ndarray]
    curriculum_columns: tuple[str, ...] = ()
    sample_positions: Optional[np.ndarray] = None

    @property
    def n_samples(self) -> int:
        """Number of extracted windows."""
        return len(self.subsequence_ids)


@dataclass(frozen=True)
class PredictionAlignment:
    """Preprocessing contract for split-end-aligned prediction groups."""

    splits: tuple[int, ...]
    prediction_length: int
    target_offset: int

    def starts(
        self, split_start: int, split_stop: int, window_length: int
    ) -> np.ndarray:
        """Return chronological, possibly negative, starts relative to a sequence."""
        count = (
            split_stop - split_start + self.prediction_length - 1
        ) // self.prediction_length
        return np.asarray(
            [
                split_stop - window_length - i * self.prediction_length
                for i in range(count - 1, -1, -1)
            ],
            dtype=np.int64,
        )


@dataclass(frozen=True)
class BatchArrays:
    """Columnar NumPy views for one ordered worker batch."""

    sequence_ids: np.ndarray
    item_positions: np.ndarray
    values: dict[str, np.ndarray]
    curriculum_values: dict[str, np.ndarray]
    split_values: Optional[np.ndarray]
    run_starts: np.ndarray
    run_stops: np.ndarray


@beartype
def _normalize_cardinality_config(
    config: Optional[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Convert validated cardinality models into serializable dictionaries."""
    normalized = {}
    for column, value in (config or {}).items():
        value = CardinalityLimitModel.model_validate(value).model_dump(mode="python")
        normalized[column] = {
            key: setting for key, setting in dict(value).items() if setting is not None
        }
    return normalized


@beartype
def _category_sort_key(value: Any) -> tuple[str, Any]:
    """Provide deterministic ordering, including for defensive mixed-type input."""
    if isinstance(value, (str, int, float, bool)):
        return type(value).__name__, value
    return type(value).__name__, repr(value)


@beartype
def _hash_id_map(num_buckets: int, start_id: int) -> dict[str, int]:
    """Create persisted, decoder-visible labels for hashing buckets."""
    result = {
        f"{HASH_BUCKET_LABEL_PREFIX}{index}]": start_id + index
        for index in range(num_buckets)
    }
    result[SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.unknown]] = (
        SPECIAL_TOKEN_IDS.unknown
    )
    result[SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.other]] = (
        SPECIAL_TOKEN_IDS.other
    )
    return result


@beartype
def _finalize_cardinality_maps(
    id_maps: dict[str, dict[Union[str, int], int]],
    value_counts: dict[str, Counter],
    cardinality_config: Mapping[str, dict[str, Any]],
) -> dict[str, dict[Union[str, int], int]]:
    """Build retained-value maps followed by optional hashing buckets."""
    for column, config in cardinality_config.items():
        retained = []
        if "min_freq" in config or "top_k" in config:
            counts = value_counts.get(column, Counter())
            if not counts:
                raise ValueError(
                    "No unmasked examples found for cardinality-controlled column "
                    f"{column!r}"
                )
            user_counts = {
                value: count
                for value, count in counts.items()
                if value not in SPECIAL_TOKEN_LABELS
            }
            if "min_freq" in config:
                retained = [
                    value
                    for value, count in user_counts.items()
                    if count >= int(config["min_freq"])
                ]
            else:
                retained = [
                    value
                    for value, _ in sorted(
                        user_counts.items(),
                        key=lambda item: (-item[1], _category_sort_key(item[0])),
                    )[: int(config["top_k"])]
                ]
        if retained:
            id_maps[column] = create_id_map(pl.DataFrame({column: retained}), column)
        else:
            id_maps[column] = {
                SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.unknown]: (
                    SPECIAL_TOKEN_IDS.unknown
                ),
                SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.other]: (
                    SPECIAL_TOKEN_IDS.other
                ),
            }
        hashing = config.get("hashing")
        if hashing is not None:
            hash_start = SPECIAL_TOKEN_IDS.user_start + len(retained)
            id_maps[column].update(
                _hash_id_map(int(hashing["num_buckets"]), hash_start)
            )
    return id_maps


@beartype
def _validate_cardinality_columns(
    cardinality_config: Mapping[str, dict[str, Any]],
    data_columns: list[str],
    precomputed_id_maps: Mapping[str, dict[Union[str, int], int]],
) -> None:
    missing = set(cardinality_config) - set(data_columns)
    if missing:
        raise ValueError(
            "cardinality_config references columns not present in the data: "
            f"{sorted(missing)}"
        )
    overlap = set(cardinality_config) & set(precomputed_id_maps)
    if overlap:
        raise ValueError(
            "cardinality_config cannot be combined with precomputed maps for the "
            f"same columns: {sorted(overlap)}"
        )


def _stable_category_hash(value: Any, seed: int) -> int:
    """Hash supported categorical scalars with Sequifier's stable FNV-1a variant."""
    if isinstance(value, (bool, np.bool_)):
        payload = b"b:1" if bool(value) else b"b:0"
    elif isinstance(value, (int, np.integer)):
        payload = f"i:{int(value)}".encode("ascii")
    elif isinstance(value, str):
        payload = b"s:" + value.encode("utf-8")
    else:
        raise TypeError(
            "Stable categorical hashing supports strings, booleans, and integers; "
            f"found {type(value).__name__}"
        )

    result = STABLE_HASH_OFFSET_BASIS ^ (seed & UINT64_MASK)
    for byte in payload:
        result ^= byte
        result = (result * STABLE_HASH_PRIME) & UINT64_MASK
    return result


@beartype
def _validate_cardinality_reserved_values(
    data: pl.DataFrame,
    cardinality_config: Mapping[str, dict[str, Any]],
) -> None:
    """Reject reserved string labels anywhere in cardinality-controlled input."""
    for column, config in cardinality_config.items():
        if column not in data.columns or not isinstance(
            data.schema[column], (pl.String, pl.Utf8, pl.Categorical)
        ):
            continue
        values = data.get_column(column).drop_nulls()
        mask_label = SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.mask]
        if values.eq(mask_label).any():
            raise ValueError(f"Found value {mask_label!r} in {column}, this is invalid")
        if config.get("hashing") is not None and any(
            _is_hash_bucket_label(value) for value in values.to_list()
        ):
            raise ValueError(
                f"Values beginning with {HASH_BUCKET_LABEL_PREFIX!r} are reserved "
                f"for hashing buckets in {column}"
            )


@beartype
def _timestamp_to_microseconds(value: Union[str, date, datetime]) -> int:
    """Convert an ISO timestamp-like cutoff to Unix microseconds."""
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"Invalid timestamp split value: {value!r}") from error
    elif isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.combine(value, datetime.min.time())
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = parsed - epoch
    return delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds


@beartype
def _normalized_split_cutoffs(
    split_values: Optional[list[Any]],
) -> Optional[np.ndarray]:
    if split_values is None:
        return None
    if all(
        isinstance(value, int) and not isinstance(value, bool) for value in split_values
    ):
        normalized = [int(value) for value in split_values]
    elif all(isinstance(value, (str, date, datetime)) for value in split_values):
        normalized = [_timestamp_to_microseconds(value) for value in split_values]
    else:
        raise ValueError(
            "split_values must contain either only integers or only timestamp values"
        )
    if normalized != sorted(normalized) or len(set(normalized)) != len(normalized):
        raise ValueError("split_values must be strictly increasing")
    return np.asarray(normalized, dtype=np.int64)


@beartype
def _split_count(
    split_ratios: Optional[list[float]], split_values: Optional[list[Any]]
) -> int:
    """Return split count for exactly one configured splitting strategy."""
    if (split_ratios is None) == (split_values is None):
        raise ValueError("Exactly one of split_ratios and split_values must be set")
    if split_ratios is not None:
        return len(split_ratios)
    assert split_values is not None
    return len(split_values) + 1


def _check_split_window_proportions(proportions: list[float]) -> None:
    """Reject an estimated training-window share smaller than one half."""
    if not proportions or proportions[0] >= 0.5:
        return
    details = ", ".join(
        f"split{i}: {proportion:.1%}" for i, proportion in enumerate(proportions)
    )
    message = (
        "Split 0 is estimated to have less than 50% of the windows written to disk: "
        f"{details}. Set ALLOW_LARGE_SPLITS=1 to allow this."
    )
    if os.environ.get("ALLOW_LARGE_SPLITS") == "1":
        warnings.warn(message, UserWarning, stacklevel=2)
    else:
        raise ValueError(message)


def _estimate_window_proportions(
    row_counts: Sequence[float],
    window_stride: int,
    alignment: Optional["PredictionAlignment"],
) -> list[float]:
    """Use each split's row share and window spacing as a cheap estimate."""
    weights = [
        count
        / (
            alignment.prediction_length
            if alignment is not None and i in alignment.splits
            else window_stride
        )
        for i, count in enumerate(row_counts)
    ]
    total = sum(weights)
    return [weight / total for weight in weights] if total else []


def _assigned_split_row_counts(
    sequence_lengths: dict[int, int],
    split_ratios: list[float],
    seed: int,
    assignments: Optional[dict[int, int]],
) -> list[int]:
    counts = [0] * len(split_ratios)
    for sequence_id, length in sequence_lengths.items():
        group = (
            assignments[sequence_id]
            if assignments is not None and sequence_id in assignments
            else assign_sequence_to_split(sequence_id, split_ratios, seed)
        )
        counts[group] += length
    return counts


def _cutoff_row_counts(values: pl.Series, cutoffs: np.ndarray) -> list[int]:
    groups = np.searchsorted(cutoffs, values.to_numpy(), side="right")
    return np.bincount(groups, minlength=len(cutoffs) + 1).tolist()


@beartype
def _require_split_ratios(
    split_ratios: Optional[list[float]],
) -> list[float]:
    """Narrow ratios at call sites used only by ratio-based split modes."""
    if split_ratios is None:
        raise ValueError("Ratio-based split methods require split_ratios")
    return split_ratios


@beartype
def _add_normalized_split_column(
    data: pl.DataFrame,
    split_column: Optional[str],
    split_values: Optional[list[Any]],
    source: str,
) -> pl.DataFrame:
    """Add one numeric comparison column while preserving the authored column."""
    if split_column is None:
        return data
    if split_column not in data.columns:
        raise ValueError(f"split_column {split_column!r} not found in {source}")
    dtype = data.schema[split_column]
    cutoffs = _normalized_split_cutoffs(split_values)
    assert cutoffs is not None
    if dtype.is_integer():
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in split_values or []
        ):
            raise ValueError("Integer split_column requires integer split_values")
        expression = pl.col(split_column).cast(pl.Int64)
    elif isinstance(dtype, (pl.Date, pl.Datetime)):
        if not all(
            isinstance(value, (str, date, datetime)) for value in split_values or []
        ):
            raise ValueError("Timestamp split_column requires timestamp split_values")
        expression = pl.col(split_column).dt.epoch("us")
    elif isinstance(dtype, (pl.String, pl.Utf8)):
        if not all(
            isinstance(value, (str, date, datetime)) for value in split_values or []
        ):
            raise ValueError("Timestamp split_column requires timestamp split_values")
        parsed = pl.col(split_column).str.to_datetime(
            time_unit="us", time_zone="UTC", strict=True
        )
        data = data.with_columns(parsed.alias(split_column))
        expression = pl.col(split_column).dt.epoch("us")
    else:
        raise ValueError(
            f"split_column must be integer or timestamp-valued; found {dtype} in {source}"
        )
    return data.with_columns(expression.alias(SPLIT_VALUE_COLUMN))


def _sequence_windows_nbytes(windows: SequenceWindows) -> int:
    arrays = [
        windows.subsequence_ids,
        windows.start_item_positions,
        windows.left_pad_lengths,
        *windows.values.values(),
    ]
    arrays.extend(
        (windows.split_start_item_positions, windows.split_end_item_positions)
    )
    if windows.sample_positions is not None:
        arrays.append(windows.sample_positions)
    return sum(array.nbytes for array in arrays)


@beartype
def _stable_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _stable_json_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return value


@beartype
def _stable_json_digest(value: Any) -> str:
    encoded = json.dumps(
        _stable_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@beartype
def _normalize_column_types(
    column_data_types: Optional[dict[str, str]],
) -> Optional[dict[str, str]]:
    if not column_data_types:
        return None
    return {
        column: (
            dtype
            if _is_temporal_dtype_name(dtype)
            else canonicalize_polars_dtype_name(dtype)
        )
        for column, dtype in column_data_types.items()
    }


@beartype
def _column_types_from_metadata(
    metadata: dict[str, Any],
) -> Optional[dict[str, str]]:
    """Read current and historical metadata field names."""
    return _normalize_column_types(
        metadata.get("column_data_types") or metadata.get("column_types")
    )


@beartype
def _configured_column_types_for_data_columns(
    column_data_types: Optional[dict[str, str]],
    data_columns: list[str],
) -> Optional[dict[str, str]]:
    if not column_data_types:
        return None

    missing_columns = [
        column for column in data_columns if column not in column_data_types
    ]
    if missing_columns:
        raise ValueError(
            "column_data_types must include every to-be-processed column. "
            f"Missing: {missing_columns}"
        )

    return {column: column_data_types[column] for column in data_columns}


@beartype
def _validate_declared_column_roles(
    data_columns: list[str],
    categorical_columns: Optional[list[str]],
    real_columns: Optional[list[str]],
) -> None:
    """Ensure explicit semantic declarations refer to processed features."""
    declared = set(categorical_columns or []) | set(real_columns or [])
    unknown = declared - set(data_columns)
    if unknown:
        raise ValueError(
            "categorical_columns and real_columns must refer to to-be-processed "
            f"columns. Unknown: {sorted(unknown)}"
        )


@beartype
def _validate_declared_roles_against_metadata(
    categorical_columns: Optional[list[str]],
    real_columns: Optional[list[str]],
    id_maps: dict[str, dict[Union[str, int], int]],
    selected_columns_statistics: dict[str, dict[str, float]],
    col_types: Optional[dict[str, str]],
) -> None:
    """Reject explicit roles that disagree with reused preprocessing metadata."""
    if col_types is None:
        return
    invalid_categorical = sorted(
        column
        for column in categorical_columns or []
        if column not in id_maps
        or column not in col_types
        or not is_integer_dtype_name(col_types[column])
    )
    invalid_real = sorted(
        column
        for column in real_columns or []
        if column not in selected_columns_statistics
        or column not in col_types
        or not is_float_dtype_name(col_types[column])
    )
    if invalid_categorical or invalid_real:
        raise ValueError(
            "Explicit column classifications disagree with metadata_config_path. "
            f"Categorical mismatches: {invalid_categorical}; "
            f"real mismatches: {invalid_real}"
        )


@beartype
def _dtype_is_numeric(dtype: Any) -> bool:
    return dtype.is_numeric() if hasattr(dtype, "is_numeric") else False


def _is_temporal_dtype_name(dtype_name: str) -> bool:
    return dtype_name.startswith(("Date", "Datetime", "Time", "Duration"))


def _torch_dtype_for_column(dtype_name: str) -> torch.dtype:
    if _is_temporal_dtype_name(dtype_name):
        return torch.int64
    return PANDAS_TO_TORCH_TYPES[dtype_name]


@beartype
def _apply_configured_input_casting(
    data: pl.DataFrame,
    data_columns: list[str],
    column_data_types: Optional[dict[str, str]],
) -> pl.DataFrame:
    """Cast input columns early when the requested type defines processing semantics."""
    configured = _configured_column_types_for_data_columns(
        column_data_types, data_columns
    )
    if configured is None:
        return data

    casts = []
    for column in data_columns:
        target_type = configured[column]
        source_dtype = data.schema[column]

        if is_float_dtype_name(target_type):
            casts.append(pl.col(column).cast(polars_dtype_from_name(target_type)))
        elif is_integer_dtype_name(target_type) and _dtype_is_numeric(source_dtype):
            casts.append(pl.col(column).cast(polars_dtype_from_name(target_type)))

    if not casts:
        return data

    return data.with_columns(casts)


@beartype
def _apply_output_type_casting(
    data: pl.DataFrame,
    data_columns: list[str],
    col_types: dict[str, str],
) -> pl.DataFrame:
    casts = [
        (
            pl.col(column).cast(pl.Int64)
            if _is_temporal_dtype_name(col_types[column])
            else pl.col(column).cast(polars_dtype_from_name(col_types[column]))
        )
        for column in data_columns
    ]
    if not casts:
        return data
    return data.with_columns(casts)


@beartype
def _highest_ranked_type(types: list[str], order: tuple[str, ...]) -> str:
    return max(types, key=lambda type_: order.index(type_))


@beartype
def _smallest_float_covering_integer_range(integer_type: str) -> str:
    integer_info = INTEGER_TYPE_INFO[integer_type]
    largest_magnitude = max(abs(int(integer_info.min)), int(integer_info.max))
    for float_type in FLOAT_TYPE_ORDER:
        if (
            largest_magnitude <= float(FLOAT_TYPE_INFO[float_type].max)
            and largest_magnitude <= FLOAT_EXACT_INTEGER_LIMITS[float_type]
        ):
            return float_type
    return "Float64"


@beartype
def _resolve_integer_sequence_type(integer_types: list[str]) -> Any:
    if not integer_types:
        raise ValueError("Cannot resolve an integer sequence type without integers")

    min_value = min(int(INTEGER_TYPE_INFO[type_].min) for type_ in integer_types)
    max_value = max(int(INTEGER_TYPE_INFO[type_].max) for type_ in integer_types)

    if min_value >= 0 and all(type_.startswith("UInt") for type_ in integer_types):
        for dtype_name in ("UInt8", "UInt16", "UInt32", "UInt64"):
            info = INTEGER_TYPE_INFO[dtype_name]
            if max_value <= int(info.max):
                return polars_dtype_from_name(dtype_name)

    for dtype_name in ("Int8", "Int16", "Int32", "Int64"):
        info = INTEGER_TYPE_INFO[dtype_name]
        if min_value >= int(info.min) and max_value <= int(info.max):
            return polars_dtype_from_name(dtype_name)

    raise ValueError(f"Cannot resolve a safe integer dtype for {integer_types}")


@beartype
def _resolve_unified_parquet_type(column_data_types: dict[str, str]) -> Any:
    if not column_data_types:
        raise ValueError("column_data_types cannot be empty")

    normalized_types = [
        (
            "Int64"
            if _is_temporal_dtype_name(type_)
            else canonicalize_polars_dtype_name(type_)
        )
        for type_ in column_data_types.values()
    ]
    float_types = [type_ for type_ in normalized_types if is_float_dtype_name(type_)]
    integer_types = [
        type_ for type_ in normalized_types if is_integer_dtype_name(type_)
    ]

    if not float_types:
        if len(set(integer_types)) > 1:
            logger.warning(
                "Multiple integer column_data_types were specified for Parquet output; "
                "using a unified integer schema."
            )
        return _resolve_integer_sequence_type(integer_types)

    resolved_float = _highest_ranked_type(float_types, FLOAT_TYPE_ORDER)
    if len(set(normalized_types)) > 1:
        logger.warning(
            "Multiple column_data_types were specified for Parquet output; "
            f"using unified sequence dtype {resolved_float}."
        )

    if integer_types:
        required_float = _highest_ranked_type(
            [
                _smallest_float_covering_integer_range(integer_type)
                for integer_type in integer_types
            ],
            FLOAT_TYPE_ORDER,
        )
        if FLOAT_TYPE_ORDER.index(required_float) > FLOAT_TYPE_ORDER.index(
            resolved_float
        ):
            logger.warning(
                "An integer column_type has a range exceeding "
                f"{resolved_float}; upgrading unified Parquet sequence dtype "
                f"to {required_float}."
            )
            resolved_float = required_float

    return polars_dtype_from_name(resolved_float)


@beartype
def _resolve_pt_extraction_type(column_data_types: dict[str, str]) -> Any:
    normalized_types = [
        (
            "Int64"
            if _is_temporal_dtype_name(type_)
            else canonicalize_polars_dtype_name(type_)
        )
        for type_ in column_data_types.values()
    ]
    if any(is_float_dtype_name(type_) for type_ in normalized_types):
        return pl.Float64
    return _resolve_integer_sequence_type(normalized_types)


@beartype
def preprocess(args: Any, args_config: dict[str, Any]) -> None:
    """Load preprocessing config and run preprocessing."""
    logger.info("--- Starting Preprocessing ---")
    config_path = args.config_path or "configs/preprocess.yaml"
    config = load_preprocessor_config(config_path, args_config)
    Preprocessor(**config.dict())
    logger.info("--- Preprocessing Complete ---")


@beartype
def _folder_input_files(data_path: str, read_format: str) -> list[str]:
    """Return folder input files in a stable traversal order."""
    file_paths = []
    for root, directories, files in os.walk(data_path):
        directories.sort()
        file_paths.extend(
            os.path.join(root, file)
            for file in sorted(files)
            if file.endswith(read_format)
        )
    if not file_paths:
        raise ValueError(f"No {read_format!r} files found under {data_path!r}.")
    return file_paths


@beartype
def _input_has_sample_positions(
    file_paths: list[str],
    read_format: str,
    curriculum_column: Optional[Union[str, list[str]]],
) -> bool:
    """Require a configured curriculum column in every input file."""
    configured_columns = curriculum_columns(curriculum_column)
    if not configured_columns:
        return False
    presence = []
    for path in file_paths:
        columns = (
            pq.read_schema(path).names
            if read_format == "parquet"
            else pl.scan_csv(path).collect_schema().names()
        )
        presence.append(all(column in columns for column in configured_columns))
    if any(presence) and not all(presence):
        raise ValueError(
            f"{curriculum_column} must be present in every input file or none"
        )
    return bool(presence and presence[0])


@beartype
def _audit_folder_sequences(
    file_paths: list[str], read_format: str, max_rows: Optional[int]
) -> tuple[list[str], set[int], list[int]]:
    """Audit folder coordinates incrementally without retaining all input rows."""
    selected_file_paths, summaries, _ = _audit_folder_sequence_summaries(
        file_paths, read_format, max_rows
    )
    fragmented = {
        sequence_id for sequence_id, summary in summaries.items() if summary[3] > 1
    }
    return selected_file_paths, fragmented, list(summaries)


@beartype
def _audit_folder_sequence_summaries(
    file_paths: list[str],
    read_format: str,
    max_rows: Optional[int],
    split_column: Optional[str] = None,
    split_values: Optional[list[Any]] = None,
) -> tuple[list[str], dict[int, list[int]], Optional[list[int]]]:
    """Return selected folder files and global sequence coordinate summaries."""
    summaries: dict[int, list[int]] = {}
    selected_file_paths = []
    rows_read = 0
    cutoffs = _normalized_split_cutoffs(split_values)
    cutoff_counts = [0] * (len(cutoffs) + 1) if cutoffs is not None else None
    for path in file_paths:
        remaining_rows = None if max_rows is None else max_rows - rows_read
        if remaining_rows is not None and remaining_rows <= 0:
            break
        columns: list[str] = list(INPUT_METADATA_COLUMNS)
        if split_column is not None:
            columns.append(split_column)
        coordinates = read_data(
            path, read_format, columns=_deduplicate_columns(columns)
        )
        if remaining_rows is not None:
            coordinates = coordinates.slice(0, remaining_rows)
        if cutoff_counts is not None and cutoffs is not None:
            coordinates = _add_normalized_split_column(
                coordinates, split_column, split_values, path
            )
            file_counts = _cutoff_row_counts(
                coordinates.get_column(SPLIT_VALUE_COLUMN), cutoffs
            )
            cutoff_counts = [
                count + additional
                for count, additional in zip(cutoff_counts, file_counts)
            ]
        coordinates = coordinates.select(list(INPUT_METADATA_COLUMNS))
        coordinates = _validate_sequence_coordinates(coordinates, path)
        selected_file_paths.append(path)
        rows_read += coordinates.height

        starts, stops = _sequence_run_bounds(coordinates)
        sequence_ids = coordinates.get_column("sequenceId").to_numpy()
        item_positions = coordinates.get_column("itemPosition").to_numpy()
        for run_start, run_stop in zip(starts, stops):
            start = int(run_start)
            stop = int(run_stop)
            sequence_id = int(sequence_ids[start])
            first_position = int(item_positions[start])
            last_position = int(item_positions[stop - 1])
            summary = summaries.get(sequence_id)
            if summary is None:
                summaries[sequence_id] = [
                    first_position,
                    last_position,
                    stop - start,
                    1,
                ]
            else:
                summary[0] = min(summary[0], first_position)
                summary[1] = max(summary[1], last_position)
                summary[2] += stop - start
                summary[3] += 1

    for sequence_id, (minimum, maximum, count, _) in summaries.items():
        if maximum - minimum + 1 != count:
            raise ValueError(
                "itemPosition must increase by one within each sequence across "
                f"folder files; sequenceId={sequence_id}, range=[{minimum}, "
                f"{maximum}], rows={count}."
            )
    return selected_file_paths, summaries, cutoff_counts


@beartype
def _fit_data_for_split_zero(
    data: pl.DataFrame,
    split_ratios: Optional[list[float]],
    split_method: str,
    seed: int,
    sequence_split_assignments: Optional[dict[int, int]] = None,
    sequence_summaries: Optional[dict[int, list[int]]] = None,
    split_values: Optional[list[Any]] = None,
) -> pl.DataFrame:
    """Select observations assigned to split 0 for metadata fitting."""
    if split_method == "value_cutoff":
        cutoffs = _normalized_split_cutoffs(split_values)
        assert cutoffs is not None
        return data.filter(pl.col(SPLIT_VALUE_COLUMN) < int(cutoffs[0]))
    if split_method == "between_sequence":
        assert split_ratios is not None
        sequence_ids = [int(value) for value in data["sequenceId"].unique()]
        training_ids = [
            sequence_id
            for sequence_id in sequence_ids
            if (
                sequence_split_assignments.get(
                    sequence_id,
                    assign_sequence_to_split(sequence_id, split_ratios, seed),
                )
                if sequence_split_assignments is not None
                else assign_sequence_to_split(sequence_id, split_ratios, seed)
            )
            == 0
        ]
        return data.filter(pl.col("sequenceId").is_in(training_ids))

    if split_method != "within_sequence":
        raise ValueError(
            "split_method must be one of 'within_sequence', 'between_sequence', "
            "'value_cutoff'"
        )

    assert split_ratios is not None

    if sequence_summaries is not None:
        training_limits = pl.DataFrame(
            {
                "sequenceId": list(sequence_summaries),
                "__fit_stop": [
                    summary[0] + int(split_ratios[0] * summary[2])
                    for summary in sequence_summaries.values()
                ],
            }
        )
        return (
            data.join(training_limits, on="sequenceId", how="left")
            .filter(pl.col("itemPosition") < pl.col("__fit_stop"))
            .drop("__fit_stop")
        )

    starts, stops = _sequence_run_bounds(data)
    training_slices = [
        data.slice(int(start), int(split_ratios[0] * (stop - start)))
        for start, stop in zip(starts, stops)
        if int(split_ratios[0] * (stop - start)) > 0
    ]
    return pl.concat(training_slices) if training_slices else data.head(0)


class Preprocessor:
    """Stateful preprocessing pipeline for single-file or folder inputs."""

    @beartype
    def __init__(
        self,
        project_root: str,
        continue_preprocessing: bool,
        preprocessing_data_path: str,
        read_format: str,
        write_format: str,
        merge_output: bool,
        allow_sequence_splitting: bool,
        selected_columns: Optional[list[str]],
        split_ratios: Optional[list[float]],
        window_length: int,
        window_stride: int,
        max_rows: Optional[int],
        seed: int,
        n_cores: Optional[int],
        batches_per_file: int,
        process_by_file: bool,
        prediction_aligned_splits: list[int],
        prediction_length: Optional[int],
        target_offset: Optional[int],
        use_precomputed_maps: Optional[list[str]],
        metadata_config_path: Optional[str],
        max_target_offset: int = 1,
        mask_column: Optional[str] = None,
        column_data_types: Optional[dict[str, str]] = None,
        categorical_columns: Optional[list[str]] = None,
        real_columns: Optional[list[str]] = None,
        split_method: str = "within_sequence",
        split_column: Optional[str] = None,
        split_values: Optional[list[Any]] = None,
        normalize_real_columns: bool = True,
        depth_layouts: Optional[dict] = None,
        curriculum_column: Optional[Union[str, list[str]]] = None,
        normalize_on_all_data: bool = False,
        cardinality_config: Optional[dict[str, Any]] = None,
    ):
        """Initialize and run preprocessing from validated config fields."""
        self.depth_layouts = DepthLayoutRegistryModel.model_validate(
            depth_layouts or {}
        )
        self.project_root = project_root
        self.batches_per_file = batches_per_file
        self.preprocessing_data_path = preprocessing_data_path
        self.read_format = read_format
        self.write_format = write_format

        self.data_name_root = os.path.splitext(
            os.path.basename(preprocessing_data_path)
        )[0]
        self.merge_output = merge_output
        if self.merge_output:
            self.target_dir = "temp"
        else:
            if write_format not in ["pt", "parquet"]:
                raise ValueError(
                    f"write_format must be 'pt' or 'parquet' when merge_output is False, got '{write_format}'"
                )
            self.target_dir = f"{self.data_name_root}-temp"

        self.allow_sequence_splitting = allow_sequence_splitting
        self.categorical_columns = categorical_columns
        self.real_columns = real_columns

        self.use_precomputed_maps = use_precomputed_maps
        self.cardinality_config = _normalize_cardinality_config(cardinality_config)
        self.metadata_config_path = metadata_config_path
        self.mask_column = mask_column
        self.curriculum_column = curriculum_column
        if split_method not in [
            "within_sequence",
            "between_sequence",
            "value_cutoff",
        ]:
            raise ValueError(
                "split_method must be one of 'within_sequence', "
                "'between_sequence', 'value_cutoff'"
            )
        self.split_method = split_method
        self.split_column = split_column
        self.split_values = split_values
        self.n_splits = _split_count(split_ratios, split_values)
        split_cutoffs = _normalized_split_cutoffs(split_values)
        if split_method == "value_cutoff":
            if not split_column or split_cutoffs is None:
                raise ValueError(
                    "value_cutoff splitting requires split_column and split_values"
                )
            if split_column == SPLIT_VALUE_COLUMN:
                raise ValueError(
                    f"split_column cannot use reserved name {SPLIT_VALUE_COLUMN!r}"
                )
            if split_column == mask_column:
                raise ValueError("split_column cannot also be mask_column")
            if split_ratios is not None:
                raise ValueError("value_cutoff splitting does not accept split_ratios")
        elif split_ratios is None:
            raise ValueError("Ratio-based split methods require split_ratios")
        self.split_ratios = split_ratios
        self.window_stride = window_stride
        self.prediction_aligned_splits = prediction_aligned_splits
        self.prediction_length = prediction_length
        self.target_offset = target_offset
        self.alignment = (
            PredictionAlignment(
                tuple(prediction_aligned_splits), prediction_length, target_offset
            )
            if prediction_aligned_splits
            and prediction_length is not None
            and target_offset is not None
            else None
        )
        if split_method == "within_sequence" and split_ratios is not None:
            _check_split_window_proportions(
                _estimate_window_proportions(
                    split_ratios, window_stride, self.alignment
                )
            )
        self.max_rows = max_rows
        self.process_by_file = process_by_file
        self.column_data_types = _normalize_column_types(column_data_types)
        self.normalize_real_columns = normalize_real_columns
        self.normalize_on_all_data = normalize_on_all_data
        self.metadata_fitted_on_all_data = normalize_on_all_data
        if self.mask_column is not None and self.metadata_config_path is None:
            raise ValueError("metadata_config_path must be set when mask_column is set")

        self.seed = seed
        np.random.seed(seed)
        self.n_cores = n_cores or multiprocessing.cpu_count()
        self.continue_preprocessing = continue_preprocessing
        self.storage_layout = StoredWindowLayout(
            window_length=window_length,
            max_target_offset=max_target_offset,
            version=CURRENT_STORED_WINDOW_LAYOUT_VERSION,
        )
        self.has_sample_positions = False
        self._setup_directories()

        if selected_columns is not None:
            selected_columns = ["sequenceId", "itemPosition"] + selected_columns
            if self.mask_column is not None and self.mask_column in selected_columns:
                raise ValueError(
                    f"'{self.mask_column}' is not allowed to be in 'selected_columns'"
                )

        self._setup_split_paths(write_format, self.n_splits)
        if self.depth_layouts:
            if (
                write_format != "pt"
                or merge_output
                or mask_column is not None
                or len(self.depth_layouts.root) != 1
            ):
                raise ValueError(
                    "Depth preprocessing requires one layout, PT output, no merge, and no mask_column"
                )
            from sequifier.io.depth_preprocess import preprocess_depth

            preprocess_depth(self, selected_columns)
            return

        if self.continue_preprocessing:
            if self.merge_output:
                paths_to_check = self.split_paths
            else:
                paths_to_check = [
                    os.path.join(
                        self.project_root, "data", f"{self.data_name_root}-split{i}"
                    )
                    for i in range(self.n_splits)
                ]

            if any(os.path.exists(p) for p in paths_to_check):
                logger.info(
                    "Existing split paths found with continue_preprocessing=True. "
                    "Skipping processing and running cleanup."
                )
                self._restore_resume_manifest_state(selected_columns, write_format)
                self._cleanup(write_format)
                return

        if os.path.isfile(preprocessing_data_path):
            data = _load_and_preprocess_data(
                preprocessing_data_path,
                read_format,
                selected_columns,
                max_rows,
                self.mask_column,
                self.curriculum_column,
                split_column=self.split_column,
                split_values=self.split_values,
            )
            self.has_sample_positions = bool(curriculum_storage_columns(data.columns))
            data_columns = _get_data_columns(data, self.mask_column)
            _validate_declared_column_roles(
                data_columns, self.categorical_columns, self.real_columns
            )
            configured_col_types = _configured_column_types_for_data_columns(
                self.column_data_types, data_columns
            )
            data = _apply_configured_input_casting(
                data, data_columns, configured_col_types
            )
            sequence_split_assignments = (
                _balanced_sequence_split_assignments(
                    data.get_column("sequenceId").unique().to_list(),
                    _require_split_ratios(split_ratios),
                    self.seed,
                )
                if self.split_method == "between_sequence"
                else None
            )
            if self.split_method == "between_sequence":
                assert split_ratios is not None
                sequence_lengths = dict(
                    data.group_by("sequenceId").agg(pl.len()).iter_rows()
                )
                _check_split_window_proportions(
                    _estimate_window_proportions(
                        _assigned_split_row_counts(
                            sequence_lengths,
                            split_ratios,
                            self.seed,
                            sequence_split_assignments,
                        ),
                        window_stride,
                        self.alignment,
                    )
                )
            if self.split_method == "value_cutoff":
                cutoffs = _normalized_split_cutoffs(self.split_values)
                assert cutoffs is not None
                _check_split_window_proportions(
                    _estimate_window_proportions(
                        _cutoff_row_counts(
                            data.get_column(SPLIT_VALUE_COLUMN), cutoffs
                        ),
                        window_stride,
                        self.alignment,
                    )
                )
            if self.metadata_config_path:
                metadata_path = os.path.join(
                    self.project_root, self.metadata_config_path
                )

                with open(metadata_path, "r") as f:
                    preexisting_metadata = json.load(f)

                self.metadata_fitted_on_all_data = preexisting_metadata.get(
                    "normalize_on_all_data", True
                )

                validate_special_token_ids(
                    preexisting_metadata["special_token_ids"],
                    source=f"metadata config '{self.metadata_config_path}'",
                )
                id_maps = preexisting_metadata["id_maps"]
                self._use_metadata_cardinality_config(preexisting_metadata)
                _validate_cardinality_columns(self.cardinality_config, data_columns, {})
                selected_columns_statistics = preexisting_metadata[
                    "selected_columns_statistics"
                ]
                n_classes = preexisting_metadata["n_classes"]
                col_types = _column_types_from_metadata(preexisting_metadata)
                _validate_declared_roles_against_metadata(
                    self.categorical_columns,
                    self.real_columns,
                    id_maps,
                    selected_columns_statistics,
                    col_types,
                )
            else:
                id_maps, selected_columns_statistics = {}, {}

                precomputed_id_maps = load_precomputed_id_maps(
                    self.project_root, data_columns, self.use_precomputed_maps
                )
                _validate_cardinality_columns(
                    self.cardinality_config, data_columns, precomputed_id_maps
                )
                _validate_cardinality_reserved_values(data, self.cardinality_config)

                fitting_data = (
                    data
                    if self.normalize_on_all_data
                    else _fit_data_for_split_zero(
                        data,
                        split_ratios,
                        self.split_method,
                        self.seed,
                        sequence_split_assignments,
                        split_values=self.split_values,
                    )
                )
                categorical_value_counts: dict[str, Counter] = {}
                id_maps, selected_columns_statistics = _get_column_statistics(
                    fitting_data,
                    data_columns,
                    id_maps,
                    selected_columns_statistics,
                    0,
                    precomputed_id_maps,
                    self.mask_column,
                    categorical_value_counts=categorical_value_counts,
                    cardinality_config=self.cardinality_config,
                    categorical_columns=self.categorical_columns,
                    real_columns=self.real_columns,
                )

                id_maps = id_maps | precomputed_id_maps
                id_maps = _finalize_cardinality_maps(
                    id_maps, categorical_value_counts, self.cardinality_config
                )
                n_classes = None
                col_types = None

            data, n_classes, col_types = _apply_column_statistics(
                data,
                data_columns,
                id_maps,
                selected_columns_statistics,
                normalize_real_columns=self.normalize_real_columns,
                n_classes=n_classes,
                col_types=configured_col_types or col_types,
                cardinality_config=self.cardinality_config,
            )
            if configured_col_types is not None:
                col_types = configured_col_types
            data = _apply_mask_column(data, data_columns, col_types, self.mask_column)
            data = _apply_output_type_casting(data, data_columns, col_types)

            self._write_or_validate_resume_manifest(
                selected_columns,
                write_format,
                data_columns,
                id_maps,
                n_classes,
                col_types,
                selected_columns_statistics,
            )
            self._export_metadata(
                id_maps, n_classes, col_types, selected_columns_statistics
            )

            schema = self._create_schema(col_types, self.storage_layout.window_length)

            n_batches = _process_batches_single_file(
                self.project_root,
                self.data_name_root,
                data,
                schema,
                self.n_cores,
                self.storage_layout,
                window_stride,
                data_columns,
                col_types,
                split_ratios,
                write_format,
                self.split_paths,
                self.target_dir,
                self.batches_per_file,
                self.merge_output,
                self.allow_sequence_splitting,
                self.split_method,
                self.seed,
                sequence_split_assignments,
                normalize_on_all_data=self.normalize_on_all_data,
                alignment=self.alignment,
                split_values=self.split_values,
            )

            if self.merge_output:
                input_files = create_file_paths_for_single_file(
                    self.project_root,
                    self.target_dir,
                    self.n_splits,
                    n_batches,
                    self.data_name_root,
                    write_format,
                )
                combine_multiprocessing_outputs(
                    self.project_root,
                    self.target_dir,
                    self.n_splits,
                    input_files,
                    self.data_name_root,
                    write_format,
                    in_target_dir=False,
                )
                delete_files(input_files)
        else:
            files_to_process = _folder_input_files(preprocessing_data_path, read_format)
            self.has_sample_positions = _input_has_sample_positions(
                files_to_process, read_format, self.curriculum_column
            )
            folder_sequence_summaries = None
            folder_cutoff_counts = None
            fragmented_sequence_ids: set[int] = set()
            folder_sequence_ids: list[int] = []
            if self.metadata_config_path or not self.normalize_on_all_data:
                (
                    files_to_process,
                    folder_sequence_summaries,
                    folder_cutoff_counts,
                ) = _audit_folder_sequence_summaries(
                    files_to_process,
                    read_format,
                    max_rows,
                    self.split_column if self.metadata_config_path else None,
                    self.split_values if self.metadata_config_path else None,
                )
                fragmented_sequence_ids = {
                    sequence_id
                    for sequence_id, summary in folder_sequence_summaries.items()
                    if summary[3] > 1
                }
                folder_sequence_ids = list(folder_sequence_summaries)

            sequence_split_assignments = (
                _balanced_sequence_split_assignments(
                    folder_sequence_ids,
                    _require_split_ratios(split_ratios),
                    self.seed,
                )
                if folder_sequence_summaries is not None
                and self.split_method == "between_sequence"
                else None
            )

            if self.metadata_config_path:
                metadata_path = os.path.join(
                    self.project_root, self.metadata_config_path
                )

                with open(metadata_path, "r") as f:
                    preexisting_metadata = json.load(f)

                self.metadata_fitted_on_all_data = preexisting_metadata.get(
                    "normalize_on_all_data", True
                )

                validate_special_token_ids(
                    preexisting_metadata["special_token_ids"],
                    source=f"metadata config '{self.metadata_config_path}'",
                )
                id_maps = preexisting_metadata["id_maps"]
                self._use_metadata_cardinality_config(preexisting_metadata)
                selected_columns_statistics = preexisting_metadata[
                    "selected_columns_statistics"
                ]
                n_classes = preexisting_metadata["n_classes"]
                col_types = _column_types_from_metadata(preexisting_metadata)

                if col_types is None:
                    raise ValueError(
                        "Metadata used with folder preprocessing must contain "
                        "'column_data_types' or the historical 'column_types' field."
                    )

                # Reconstruct data_columns from the provided col_types
                data_columns = [
                    col
                    for col in col_types.keys()
                    if col not in _reserved_input_columns(self.mask_column)
                ]
                _validate_cardinality_columns(self.cardinality_config, data_columns, {})
                configured_col_types = _configured_column_types_for_data_columns(
                    self.column_data_types, data_columns
                )
                _validate_declared_column_roles(
                    data_columns, self.categorical_columns, self.real_columns
                )
                _validate_declared_roles_against_metadata(
                    self.categorical_columns,
                    self.real_columns,
                    id_maps,
                    selected_columns_statistics,
                    col_types,
                )
                if configured_col_types is not None:
                    col_types = configured_col_types

            else:
                (
                    files_to_process,
                    n_classes,
                    id_maps,
                    selected_columns_statistics,
                    col_types,
                    data_columns,
                    fragmented_sequence_ids,
                    folder_sequence_summaries,
                    folder_cutoff_counts,
                ) = self._get_column_metadata_across_files(
                    preprocessing_data_path,
                    read_format,
                    max_rows,
                    selected_columns,
                    self.column_data_types,
                    files_to_process,
                    sequence_split_assignments,
                    folder_sequence_summaries,
                )
                folder_sequence_ids = list(folder_sequence_summaries)
                for col in id_maps:
                    if self.column_data_types is None:
                        col_types[col] = "Int64"

            sequence_split_assignments = sequence_split_assignments or (
                _balanced_sequence_split_assignments(
                    folder_sequence_ids,
                    _require_split_ratios(split_ratios),
                    self.seed,
                )
                if self.split_method == "between_sequence"
                else None
            )

            if self.split_method == "between_sequence":
                assert (
                    split_ratios is not None and folder_sequence_summaries is not None
                )
                sequence_lengths = {
                    sequence_id: summary[2]
                    for sequence_id, summary in folder_sequence_summaries.items()
                }
                _check_split_window_proportions(
                    _estimate_window_proportions(
                        _assigned_split_row_counts(
                            sequence_lengths,
                            split_ratios,
                            self.seed,
                            sequence_split_assignments,
                        ),
                        window_stride,
                        self.alignment,
                    )
                )
            if self.split_method == "value_cutoff":
                assert folder_cutoff_counts is not None
                _check_split_window_proportions(
                    _estimate_window_proportions(
                        folder_cutoff_counts,
                        window_stride,
                        self.alignment,
                    )
                )

            self._write_or_validate_resume_manifest(
                selected_columns,
                write_format,
                data_columns,
                id_maps,
                n_classes,
                col_types,
                selected_columns_statistics,
            )
            self._export_metadata(
                id_maps, n_classes, col_types, selected_columns_statistics
            )
            schema = self._create_schema(col_types, self.storage_layout.window_length)

            if fragmented_sequence_ids:
                logger.warning(
                    "Sequences span multiple input files; processing the folder as "
                    "one logical dataset to preserve sequence boundaries."
                )
                self._process_fragmented_folder(
                    files_to_process,
                    read_format,
                    selected_columns,
                    max_rows,
                    schema,
                    self.storage_layout,
                    window_stride,
                    data_columns,
                    n_classes,
                    id_maps,
                    selected_columns_statistics,
                    col_types,
                    split_ratios,
                    write_format,
                    sequence_split_assignments,
                )
            else:
                self._process_batches_multiple_files(
                    files_to_process,
                    read_format,
                    selected_columns,
                    max_rows,
                    schema,
                    self.n_cores,
                    self.storage_layout,
                    window_stride,
                    data_columns,
                    n_classes,
                    id_maps,
                    selected_columns_statistics,
                    col_types,
                    split_ratios,
                    write_format,
                    process_by_file,
                    sequence_split_assignments=sequence_split_assignments,
                )

        self._cleanup(write_format)

    @beartype
    def _use_metadata_cardinality_config(self, metadata: Mapping[str, Any]) -> None:
        """Use the fitted transform persisted with reused preprocessing metadata."""
        persisted = _normalize_cardinality_config(metadata.get("cardinality_config"))
        if self.cardinality_config and self.cardinality_config != persisted:
            raise ValueError(
                "cardinality_config must match the configuration in "
                "metadata_config_path"
            )
        self.cardinality_config = persisted

    @beartype
    def _create_schema(
        self, col_types: dict[str, str], window_length: int
    ) -> dict[str, Any]:
        """Build the long-format extracted-window schema."""
        schema: dict[str, Any] = {
            "sequenceId": pl.Int64,
            "subsequenceId": pl.Int64,
            "startItemPosition": pl.Int64,
            "leftPadLength": pl.Int64,
            "splitStartItemPosition": pl.Int64,
            "splitEndItemPosition": pl.Int64,
        }
        if self.has_sample_positions:
            schema.update(
                {
                    curriculum_storage_column(column): pl.Int64
                    for column in curriculum_columns(self.curriculum_column)
                }
            )
        schema["inputCol"] = pl.String

        if self.write_format == "parquet":
            sequence_position_type = _resolve_unified_parquet_type(col_types)
        else:
            sequence_position_type = _resolve_pt_extraction_type(col_types)

        schema.update(
            {str(i): sequence_position_type for i in range(window_length - 1, -1, -1)}
        )

        return schema

    @beartype
    def _get_column_metadata_across_files(
        self,
        data_path: str,
        read_format: str,
        max_rows: Optional[int],
        selected_columns: Optional[list[str]],
        column_data_types: Optional[dict[str, str]],
        files_to_process: list[str],
        sequence_split_assignments: Optional[dict[int, int]] = None,
        sequence_summaries: Optional[dict[int, list[int]]] = None,
    ) -> tuple[
        list[str],
        dict[str, int],
        dict[str, dict[Union[str, int], int]],
        dict[str, dict[str, float]],
        dict[str, str],
        list[str],
        set[int],
        dict[int, list[int]],
        Optional[list[int]],
    ]:
        """Accumulate metadata, statistics, and sequence audit in one file pass."""

        n_rows_running_count = 0
        id_maps, selected_columns_statistics = {}, {}
        col_types, data_columns = None, None
        sample_position_presence: bool | None = None
        selected_file_paths = []
        coordinate_summaries: dict[int, list[int]] = {}
        cutoffs = (
            _normalized_split_cutoffs(self.split_values)
            if self.split_method == "value_cutoff"
            else None
        )
        cutoff_counts = [0] * (len(cutoffs) + 1) if cutoffs is not None else None
        categorical_value_sets: dict[str, set[Any]] = {}
        categorical_value_counts: dict[str, Counter] = {}

        precomputed_id_maps = load_precomputed_id_maps(
            self.project_root, data_columns, self.use_precomputed_maps
        )

        logger.info(f"Data path: {data_path}")
        for path in files_to_process:
            if max_rows is not None and n_rows_running_count >= max_rows:
                break
            file = os.path.basename(path)
            logger.info(f"Preprocessing: reading {file}")
            max_rows_inner = (
                None if max_rows is None else max(0, max_rows - n_rows_running_count)
            )
            data = _load_and_preprocess_data(
                path,
                read_format,
                selected_columns,
                max_rows_inner,
                self.mask_column,
                self.curriculum_column,
                False,
                self.split_column,
                self.split_values,
            )
            if cutoff_counts is not None and cutoffs is not None:
                file_counts = _cutoff_row_counts(
                    data.get_column(SPLIT_VALUE_COLUMN), cutoffs
                )
                cutoff_counts = [
                    count + additional
                    for count, additional in zip(cutoff_counts, file_counts)
                ]
            selected_file_paths.append(path)
            file_summaries = data.group_by("sequenceId").agg(
                pl.col("itemPosition").min().alias("__minimum"),
                pl.col("itemPosition").max().alias("__maximum"),
                pl.len().alias("__count"),
            )
            for (
                sequence_id,
                first_position,
                last_position,
                count,
            ) in file_summaries.iter_rows():
                sequence_id = int(sequence_id)
                first_position = int(first_position)
                last_position = int(last_position)
                count = int(count)
                summary = coordinate_summaries.get(sequence_id)
                if summary is None:
                    coordinate_summaries[sequence_id] = [
                        first_position,
                        last_position,
                        count,
                        1,
                    ]
                else:
                    summary[0] = min(summary[0], first_position)
                    summary[1] = max(summary[1], last_position)
                    summary[2] += count
                    summary[3] += 1
            current_has_positions = bool(curriculum_storage_columns(data.columns))
            if sample_position_presence is None:
                sample_position_presence = current_has_positions
            elif sample_position_presence != current_has_positions:
                raise ValueError(
                    "The curriculum column must be present in every input file or none"
                )

            current_file_cols = _get_data_columns(data, self.mask_column)
            _validate_declared_column_roles(
                current_file_cols, self.categorical_columns, self.real_columns
            )
            current_configured_col_types = _configured_column_types_for_data_columns(
                column_data_types, current_file_cols
            )
            data = _apply_configured_input_casting(
                data, current_file_cols, current_configured_col_types
            )

            if col_types is None:
                data_columns = current_file_cols
                col_types = current_configured_col_types or {
                    col: str(data.schema[col]) for col in data_columns
                }
                for col in precomputed_id_maps:
                    if col not in data_columns:
                        raise ValueError(
                            f"Precomputed column {col} not found in {file}"
                        )
            else:
                if set(current_file_cols) != set(col_types):
                    missing = set(col_types) - set(current_file_cols)
                    extra = set(current_file_cols) - set(col_types)
                    raise ValueError(
                        f"Schema mismatch in file '{file}'.\n"
                        f"Expected columns: {list(col_types)}\n"
                        f"Found columns: {current_file_cols}\n"
                        f"Missing: {missing}\n"
                        f"Extra: {extra}"
                    )
                if column_data_types is None:
                    for col in current_file_cols:
                        if str(data.schema[col]) != col_types[col]:
                            raise ValueError(
                                f"Type mismatch for column '{col}' in file '{file}'. "
                                f"Expected {col_types[col]}, got "
                                f"{str(data.schema[col])}"
                            )

            if data_columns is None:
                raise ValueError("data_columns is None")
            _validate_cardinality_columns(
                self.cardinality_config, data_columns, precomputed_id_maps
            )
            _validate_cardinality_reserved_values(data, self.cardinality_config)

            fitting_data = (
                data
                if self.normalize_on_all_data
                else _fit_data_for_split_zero(
                    data,
                    self.split_ratios,
                    self.split_method,
                    self.seed,
                    sequence_split_assignments,
                    sequence_summaries,
                    self.split_values,
                )
            )
            id_maps, selected_columns_statistics = _get_column_statistics(
                fitting_data,
                data_columns,
                id_maps,
                selected_columns_statistics,
                n_rows_running_count,
                precomputed_id_maps,
                self.mask_column,
                categorical_value_sets,
                categorical_columns=self.categorical_columns,
                real_columns=self.real_columns,
                categorical_value_counts=categorical_value_counts,
                cardinality_config=self.cardinality_config,
            )
            n_rows_running_count += data.height

        fragmented_sequence_ids = {
            sequence_id
            for sequence_id, summary in coordinate_summaries.items()
            if summary[3] > 1
        }
        for sequence_id, (minimum, maximum, count, _) in coordinate_summaries.items():
            if maximum - minimum + 1 != count:
                raise ValueError(
                    "itemPosition must increase by one within each sequence across "
                    f"folder files; sequenceId={sequence_id}, range=[{minimum}, "
                    f"{maximum}], rows={count}."
                )

        for column, values in categorical_value_sets.items():
            id_maps[column] = create_id_map(
                pl.DataFrame({column: list(values)}), column
            )
        id_maps = id_maps | precomputed_id_maps
        id_maps = _finalize_cardinality_maps(
            id_maps, categorical_value_counts, self.cardinality_config
        )

        if data_columns is None:
            raise RuntimeError("data_columns was not initialized correctly.")
        n_classes = {
            col: max(SPECIAL_TOKEN_IDS.user_start, max(id_maps[col].values()) + 1)
            for col in id_maps
        }

        if col_types is None:
            raise RuntimeError("col_types was not initialized correctly.")
        if column_data_types is None:
            for column in id_maps:
                col_types[column] = "Int64"
            for column in selected_columns_statistics:
                if not is_float_dtype_name(col_types[column]):
                    col_types[column] = "Float64"
        self.has_sample_positions = bool(sample_position_presence)
        return (
            selected_file_paths,
            n_classes,
            id_maps,
            selected_columns_statistics,
            col_types,
            data_columns,
            fragmented_sequence_ids,
            coordinate_summaries,
            cutoff_counts,
        )

    @beartype
    def _setup_directories(self) -> None:
        """Prepare or validate preprocessing temp directories."""

        temp_path = os.path.join(self.project_root, "data", self.target_dir)

        if self.continue_preprocessing:
            if not os.path.exists(temp_path):
                raise Exception(f"temp folder at '{temp_path}' does not exist")
        else:
            os.makedirs(os.path.join(self.project_root, "data"), exist_ok=True)
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
            os.makedirs(temp_path)

    @beartype
    def _setup_split_paths(self, write_format: str, n_splits: int) -> None:
        """Set final split output paths."""
        split_paths = [
            os.path.join(
                self.project_root,
                "data",
                f"{self.data_name_root}-split{i}.{write_format}",
            )
            for i in range(n_splits)
        ]

        self.split_paths = split_paths

    @beartype
    def _process_fragmented_folder(
        self,
        file_paths: list[str],
        read_format: str,
        selected_columns: Optional[list[str]],
        max_rows: Optional[int],
        schema: Any,
        layout: StoredWindowLayout,
        window_stride: int,
        data_columns: list[str],
        n_classes: dict[str, int],
        id_maps: dict[str, dict[Union[int, str], int]],
        selected_columns_statistics: dict[str, dict[str, float]],
        col_types: dict[str, str],
        split_ratios: Optional[list[float]],
        write_format: str,
        sequence_split_assignments: Optional[dict[int, int]],
    ) -> None:
        """Process cross-file fragments through bounded disk-backed hash buckets."""
        total_input_bytes = sum(os.path.getsize(path) for path in file_paths)
        target_bucket_bytes = 256 * 1024 * 1024
        bucket_count = min(
            4096,
            max(self.n_cores, math.ceil(total_input_bytes / target_bucket_bytes)),
        )
        bucket_dir = os.path.join(
            self.project_root, "data", self.target_dir, "fragment-buckets"
        )
        os.makedirs(bucket_dir, exist_ok=True)
        bucket_files: dict[int, list[str]] = {}
        rows_read = 0

        for file_index, path in enumerate(file_paths):
            remaining_rows = None if max_rows is None else max_rows - rows_read
            if remaining_rows is not None and remaining_rows <= 0:
                break
            data = _load_and_preprocess_data(
                path,
                read_format,
                selected_columns,
                remaining_rows,
                self.mask_column,
                self.curriculum_column,
                False,
                self.split_column,
                self.split_values,
            )
            data = _apply_configured_input_casting(data, data_columns, col_types)
            data, _, _ = _apply_column_statistics(
                data,
                data_columns,
                id_maps,
                selected_columns_statistics,
                self.normalize_real_columns,
                n_classes,
                col_types,
                self.cardinality_config,
            )
            data = _apply_mask_column(data, data_columns, col_types, self.mask_column)
            data = _apply_output_type_casting(data, data_columns, col_types)
            data = data.with_columns(
                (
                    pl.col("sequenceId").hash(seed=self.seed % (2**64))
                    % pl.lit(bucket_count)
                ).alias("__fragment_bucket")
            )
            for partition in data.partition_by(
                "__fragment_bucket", maintain_order=False
            ):
                bucket = int(partition.get_column("__fragment_bucket")[0])
                bucket_path = os.path.join(
                    bucket_dir, f"bucket-{bucket}-{file_index}.parquet"
                )
                partition.drop("__fragment_bucket").write_parquet(
                    bucket_path, compression="lz4"
                )
                bucket_files.setdefault(bucket, []).append(bucket_path)
            rows_read += data.height

        merged_bucket_outputs: dict[int, list[str]] = {
            split: [] for split in range(self.n_splits)
        }
        worker_pool = (
            _create_preprocess_pool(self.n_cores) if self.n_cores > 1 else None
        )
        try:
            for bucket, paths in sorted(bucket_files.items()):
                data = _validate_sequence_coordinates(
                    pl.concat([pl.read_parquet(path) for path in paths]),
                    f"fragment bucket {bucket}",
                )
                bucket_name = f"{self.data_name_root}-fragment-{bucket}"
                bucket_split_paths = [
                    str(
                        Path(path).with_name(
                            Path(path).name.replace(self.data_name_root, bucket_name, 1)
                        )
                    )
                    for path in self.split_paths
                ]
                n_batches = _process_batches_single_file(
                    self.project_root,
                    bucket_name,
                    data,
                    schema,
                    self.n_cores,
                    layout,
                    window_stride,
                    data_columns,
                    col_types,
                    split_ratios,
                    write_format,
                    bucket_split_paths,
                    self.target_dir,
                    self.batches_per_file,
                    self.merge_output,
                    self.allow_sequence_splitting,
                    self.split_method,
                    self.seed,
                    sequence_split_assignments,
                    worker_pool,
                    normalize_on_all_data=self.normalize_on_all_data,
                    alignment=self.alignment,
                    split_values=self.split_values,
                )
                if self.merge_output:
                    worker_files = create_file_paths_for_single_file(
                        self.project_root,
                        self.target_dir,
                        self.n_splits,
                        n_batches,
                        bucket_name,
                        write_format,
                    )
                    combine_multiprocessing_outputs(
                        self.project_root,
                        self.target_dir,
                        self.n_splits,
                        worker_files,
                        bucket_name,
                        write_format,
                        in_target_dir=True,
                    )
                    delete_files(worker_files)
                    for split in range(self.n_splits):
                        merged_bucket_outputs[split].append(
                            create_split_file_path(
                                self.project_root,
                                bucket_name,
                                split,
                                write_format,
                                True,
                                self.target_dir,
                                None,
                                None,
                            )
                        )
        finally:
            if worker_pool is not None:
                worker_pool.close()
                worker_pool.join()
            shutil.rmtree(bucket_dir)

        if self.merge_output:
            combine_multiprocessing_outputs(
                self.project_root,
                self.target_dir,
                self.n_splits,
                merged_bucket_outputs,
                self.data_name_root,
                write_format,
                in_target_dir=False,
            )
            delete_files(merged_bucket_outputs)

    @beartype
    def _process_batches_multiple_files(
        self,
        file_paths: list[str],
        read_format: str,
        selected_columns: Optional[list[str]],
        max_rows: Optional[int],
        schema: Any,
        n_cores: int,
        layout: StoredWindowLayout,
        window_stride: int,
        data_columns: list[str],
        n_classes: dict[str, int],
        id_maps: dict[str, dict[Union[int, str], int]],
        selected_columns_statistics: dict[str, dict[str, float]],
        col_types: dict[str, str],
        split_ratios: Optional[list[float]],
        write_format: str,
        process_by_file: bool = True,
        mask_column: Optional[str] = None,
        sequence_split_assignments: Optional[dict[int, int]] = None,
    ) -> None:
        """Dispatch folder preprocessing by file or by process shard."""
        if mask_column is None:
            mask_column = self.mask_column

        if not process_by_file and max_rows is not None:
            logger.warning(
                "process_by_file=False cannot preserve a folder-global max_rows "
                "limit across file shards; processing files serially instead."
            )
            process_by_file = True

        if process_by_file:
            _process_batches_multiple_files_inner(
                project_root=self.project_root,
                data_name_root=self.data_name_root,
                process_id=0,
                file_paths=file_paths,
                read_format=read_format,
                selected_columns=selected_columns,
                max_rows=max_rows,
                schema=schema,
                n_cores=n_cores,
                layout=layout,
                window_stride=window_stride,
                data_columns=data_columns,
                n_classes=n_classes,
                id_maps=id_maps,
                selected_columns_statistics=selected_columns_statistics,
                col_types=col_types,
                split_ratios=split_ratios,
                write_format=write_format,
                split_paths=self.split_paths,
                target_dir=self.target_dir,
                batches_per_file=self.batches_per_file,
                merge_output=self.merge_output,
                allow_sequence_splitting=self.allow_sequence_splitting,
                continue_preprocessing=self.continue_preprocessing,
                mask_column=mask_column,
                split_method=self.split_method,
                seed=self.seed,
                normalize_real_columns=self.normalize_real_columns,
                sequence_split_assignments=sequence_split_assignments,
                curriculum_column=self.curriculum_column,
                split_column=self.split_column,
                split_values=self.split_values,
                normalize_on_all_data=self.normalize_on_all_data,
                alignment=self.alignment,
                cardinality_config=self.cardinality_config,
            )
            input_files = create_file_paths_for_multiple_files2(
                self.project_root,
                self.target_dir,
                self.n_splits,
                1,
                {0: len(file_paths)},
                self.data_name_root,
                write_format,
            )
        else:
            assert process_by_file is False
            worker_count = min(n_cores, len(file_paths))
            file_sizes = [os.path.getsize(path) for path in file_paths]
            file_sets = []
            next_file = 0
            remaining_bytes = sum(file_sizes)
            for worker_index in range(worker_count):
                remaining_workers = worker_count - worker_index
                target_bytes = math.ceil(remaining_bytes / remaining_workers)
                file_set = []
                file_set_bytes = 0
                while next_file < len(file_paths):
                    files_remaining = len(file_paths) - next_file
                    if (
                        file_set
                        and remaining_workers > 1
                        and (
                            file_set_bytes >= target_bytes
                            or files_remaining == remaining_workers - 1
                        )
                    ):
                        break
                    file_set.append(file_paths[next_file])
                    file_set_bytes += file_sizes[next_file]
                    next_file += 1
                file_sets.append(file_set)
                remaining_bytes -= file_set_bytes

            kwargs_1 = {
                "project_root": self.project_root,
                "data_name_root": self.data_name_root,
            }
            kwargs_2 = {
                "read_format": read_format,
                "selected_columns": selected_columns,
                "max_rows": max_rows,
                "schema": schema,
                "n_cores": 1,
                "layout": layout,
                "window_stride": window_stride,
                "data_columns": data_columns,
                "n_classes": n_classes,
                "id_maps": id_maps,
                "selected_columns_statistics": selected_columns_statistics,
                "col_types": col_types,
                "split_ratios": split_ratios,
                "write_format": write_format,
                "split_paths": self.split_paths,
                "target_dir": self.target_dir,
                "batches_per_file": self.batches_per_file,
                "merge_output": self.merge_output,
                "allow_sequence_splitting": self.allow_sequence_splitting,
                "continue_preprocessing": self.continue_preprocessing,
                "mask_column": mask_column,
                "split_method": self.split_method,
                "seed": self.seed,
                "normalize_real_columns": self.normalize_real_columns,
                "sequence_split_assignments": sequence_split_assignments,
                "curriculum_column": getattr(self, "curriculum_column", None),
                "split_column": self.split_column,
                "split_values": self.split_values,
                "normalize_on_all_data": self.normalize_on_all_data,
                "alignment": self.alignment,
                "cardinality_config": self.cardinality_config,
            }

            job_params = [
                list(kwargs_1.values())
                + [process_id, file_set]
                + list(kwargs_2.values())
                for process_id, file_set in enumerate(file_sets)
            ]
            logger.info(f"_process_batches_multiple_files n_cores: {n_cores}")
            logger.info(f"_process_batches_multiple_files {len(job_params) = }")

            with _create_preprocess_pool(len(job_params)) as pool:
                pool.starmap(_process_batches_multiple_files_inner, job_params)

            input_files = create_file_paths_for_multiple_files2(
                self.project_root,
                self.target_dir,
                self.n_splits,
                len(job_params),
                {i: len(file_sets[i]) for i in range(len(file_sets))},
                self.data_name_root,
                write_format,
            )
        if self.merge_output:
            combine_multiprocessing_outputs(
                self.project_root,
                self.target_dir,
                self.n_splits,
                input_files,
                self.data_name_root,
                write_format,
                in_target_dir=False,
            )
            delete_files(input_files)

    @beartype
    def _cleanup(self, write_format: str) -> None:
        """Move split outputs, write folder metadata, and remove temp files."""

        logger.info("Start cleanup")
        temp_output_path = os.path.join(self.project_root, "data", self.target_dir)
        directory = Path(temp_output_path)

        if not self.target_dir == "temp":
            for i, split_path in enumerate(self.split_paths):
                split = f"split{i}"
                folder_path = os.path.join(
                    self.project_root, "data", f"{self.data_name_root}-{split}"
                )
                if folder_path not in split_path:
                    raise ValueError(
                        f"Folder path '{folder_path}' mismatch with split path '{split_path}'"
                    )

                logger.info(f"Make path '{folder_path}'")
                os.makedirs(folder_path, exist_ok=True)

                pattern = re.compile(rf".+split{i}-\d+-\d+\.{re.escape(write_format)}")

                for file_path in directory.iterdir():
                    if file_path.is_file() and pattern.fullmatch(file_path.name):
                        destination = Path(folder_path) / file_path.name
                        logger.info(f"Moving '{file_path}' to '{destination}'")
                        shutil.move(str(file_path), str(destination))
                        summary_path = Path(f"{file_path}.metadata.json")
                        if summary_path.exists():
                            shutil.move(
                                str(summary_path),
                                str(Path(f"{destination}.metadata.json")),
                            )

                self._create_metadata_for_folder(folder_path, write_format, i)

        if not os.listdir(directory) or self.target_dir == "temp":
            shutil.rmtree(directory)

    @beartype
    def _layout_metadata(self) -> dict[str, Any]:
        return {
            "depth_layouts": self.depth_layouts.model_dump(mode="json"),
            "tensor_payload_version": 2 if self.depth_layouts else 1,
            "window_length": self.storage_layout.window_length,
            "max_target_offset": self.storage_layout.max_target_offset,
            "stored_window_layout_version": self.storage_layout.version,
            "prediction_aligned_splits": self.prediction_aligned_splits,
            "prediction_length": self.prediction_length,
            "target_offset": self.target_offset,
        }

    @beartype
    def _write_or_validate_resume_manifest(
        self,
        selected_columns: Optional[list[str]],
        write_format: str,
        data_columns: list[str],
        id_maps: dict[str, dict[Union[str, int], int]],
        n_classes: dict[str, int],
        col_types: dict[str, str],
        selected_columns_statistics: dict[str, dict[str, float]],
    ) -> None:
        manifest = {
            "manifest_version": 1,
            "preprocessing_config": {
                **self._layout_metadata(),
                "read_format": self.read_format,
                "write_format": write_format,
                "merge_output": self.merge_output,
                "selected_columns": selected_columns,
                "data_columns": data_columns,
                "categorical_columns": self.categorical_columns,
                "real_columns": self.real_columns,
                "split_ratios": self.split_ratios,
                "split_method": self.split_method,
                "split_column": self.split_column,
                "split_values": (
                    [
                        (
                            value.isoformat()
                            if isinstance(value, (date, datetime))
                            else value
                        )
                        for value in self.split_values
                    ]
                    if self.split_values is not None
                    else None
                ),
                "seed": self.seed,
                "window_stride": self.window_stride,
                "max_rows": self.max_rows,
                "process_by_file": self.process_by_file,
                "prediction_aligned_splits": self.prediction_aligned_splits,
                "mask_column": self.mask_column,
                **(
                    {"curriculum_column": self.curriculum_column}
                    if getattr(self, "curriculum_column", None) is not None
                    else {}
                ),
                "use_precomputed_maps": self.use_precomputed_maps,
                "cardinality_config": getattr(self, "cardinality_config", {}),
                "n_classes": n_classes,
                "id_maps": id_maps,
                "column_data_types": col_types,
                "selected_columns_statistics": selected_columns_statistics,
                "normalize_real_columns": self.normalize_real_columns,
                "normalize_on_all_data": self.normalize_on_all_data,
                "special_token_ids": SPECIAL_TOKEN_IDS.ids_by_label,
            },
        }
        manifest_path = os.path.join(
            self.project_root, "data", self.target_dir, "preprocess-manifest.json"
        )

        with open(
            os.path.join(
                self.project_root,
                "data",
                self.target_dir,
                "preprocess-manifest-check.json",
            ),
            "w",
        ) as f:
            json.dump(manifest, f, indent=4)

        if self.continue_preprocessing:
            if not os.path.exists(manifest_path):
                raise ValueError(
                    "Cannot continue preprocessing because the temp manifest is missing."
                )
            with open(manifest_path, "r") as f:
                previous_manifest = json.load(f)
            previous_manifest.get("preprocessing_config", {}).setdefault(
                "depth_layouts", {}
            )
            previous_manifest.get("preprocessing_config", {}).setdefault(
                "tensor_payload_version", 1
            )
            previous_manifest.get("preprocessing_config", {}).setdefault(
                "normalize_on_all_data", True
            )
            previous_manifest.get("preprocessing_config", {}).setdefault(
                "categorical_columns", None
            )
            previous_manifest.get("preprocessing_config", {}).setdefault(
                "real_columns", None
            )
            previous_manifest.get("preprocessing_config", {}).setdefault(
                "cardinality_config", {}
            )
            if _stable_json_value(previous_manifest) != _stable_json_value(manifest):
                raise ValueError(
                    "Cannot continue preprocessing with a different preprocessing "
                    "manifest. Check sequence layout, input path, selected/data "
                    "columns, output format, mask/metadata settings, split/stride "
                    "settings, max_rows, process_by_file, prediction alignment, "
                    "or metadata/maps/statistics."
                )
            return

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)

    @beartype
    def _restore_resume_manifest_state(
        self,
        selected_columns: Optional[list[str]],
        write_format: str,
    ) -> None:
        """Validate a completed resume and restore metadata-generation state."""
        manifest_path = os.path.join(
            self.project_root, "data", self.target_dir, "preprocess-manifest.json"
        )
        if not os.path.exists(manifest_path):
            raise ValueError(
                "Cannot continue preprocessing because the temp manifest is missing."
            )
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        previous = manifest.get("preprocessing_config", {})
        required = {
            "data_columns",
            "id_maps",
            "n_classes",
            "column_data_types",
            "selected_columns_statistics",
        }
        missing = required - set(previous)
        if missing:
            raise ValueError(
                "Cannot continue preprocessing because the temp manifest is missing "
                f"required state: {sorted(missing)}."
            )

        self._write_or_validate_resume_manifest(
            selected_columns,
            write_format,
            previous["data_columns"],
            previous["id_maps"],
            previous["n_classes"],
            previous["column_data_types"],
            previous["selected_columns_statistics"],
        )
        self.output_n_classes = previous["n_classes"]
        self.output_column_data_types = previous["column_data_types"]
        self.has_sample_positions = bool(
            curriculum_columns(previous.get("curriculum_column"))
        )

    @beartype
    def _export_metadata(
        self,
        id_maps: dict[str, dict[Union[str, int], int]],
        n_classes: dict[str, int],
        col_types: dict[str, str],
        selected_columns_statistics: dict[str, dict[str, float]],
    ) -> None:
        """Write metadata config JSON for training/inference."""
        self.output_n_classes = n_classes
        self.output_column_data_types = col_types
        data_driven_config = {
            "n_classes": n_classes,
            "id_maps": id_maps,
            "cardinality_config": self.cardinality_config,
            "special_token_ids": SPECIAL_TOKEN_IDS.ids_by_label,
            "split_paths": [
                os.path.splitext(split_path)[0] if not self.merge_output else split_path
                for split_path in self.split_paths
            ],
            "column_data_types": col_types,
            "selected_columns_statistics": {
                col: {"mean": stats["mean"], "std": stats["std"]}
                for col, stats in selected_columns_statistics.items()
            },
            "normalize_real_columns": self.normalize_real_columns,
            "normalize_on_all_data": self.metadata_fitted_on_all_data,
            "window_stride": self.window_stride,
            "prediction_aligned_splits": self.prediction_aligned_splits,
            "split_ratios": self.split_ratios,
            "split_method": self.split_method,
            "split_column": self.split_column,
            "split_values": (
                [
                    value.isoformat() if isinstance(value, (date, datetime)) else value
                    for value in self.split_values
                ]
                if self.split_values is not None
                else None
            ),
            **self._layout_metadata(),
        }
        os.makedirs(
            os.path.join(self.project_root, "configs", "metadata_configs"),
            exist_ok=True,
        )

        with open(
            os.path.join(
                self.project_root,
                "configs",
                "metadata_configs",
                f"{self.data_name_root}.json",
            ),
            "w",
        ) as f:
            json.dump(data_driven_config, f)

    @beartype
    def _create_metadata_for_folder(
        self, folder_path: str, write_format: str, split_index: int
    ) -> None:
        """Write metadata.json for an unmerged split folder."""
        logger.info(f"Creating metadata for folder '{folder_path}'")
        batch_files_metadata = []
        total_samples = 0
        directory = Path(folder_path)

        # Find files matching the current write_format
        files = sorted(
            [
                f
                for f in directory.iterdir()
                if f.is_file() and f.suffix == f".{write_format}"
            ]
        )

        for file_path in files:
            try:
                summary_path = Path(f"{file_path}.metadata.json")
                if summary_path.exists():
                    with open(summary_path, "r") as summary_file:
                        summary = json.load(summary_file)
                    batch_files_metadata.append(
                        {
                            "path": file_path.name,
                            "samples": int(summary["samples"]),
                            "left_pad_length_histogram": summary[
                                "left_pad_length_histogram"
                            ],
                            "target_valid_from_histogram": summary[
                                "target_valid_from_histogram"
                            ],
                        }
                    )
                    total_samples += int(summary["samples"])
                    os.remove(summary_path)
                    continue
                if write_format == "pt":
                    payload = load_pt_payload(
                        file_path,
                        layouts=self.depth_layouts,
                        n_classes=getattr(self, "output_n_classes", None),
                    )
                    sequences_dict = payload.sequences
                    left_pad_lengths = payload.left_pad_lengths
                    target_valid_from = torch.maximum(
                        left_pad_lengths,
                        payload.split_start_item_positions
                        - payload.start_item_positions,
                    )
                    if sequences_dict:
                        n_samples = sequences_dict[
                            list(sequences_dict.keys())[0]
                        ].shape[0]
                        batch_files_metadata.append(
                            {
                                "path": file_path.name,
                                "samples": n_samples,
                                "left_pad_length_histogram": {
                                    str(value): count
                                    for value, count in Counter(
                                        left_pad_lengths.tolist()
                                    ).items()
                                },
                                "target_valid_from_histogram": {
                                    str(value): count
                                    for value, count in Counter(
                                        target_valid_from.tolist()
                                    ).items()
                                },
                            }
                        )
                        total_samples += n_samples
                elif write_format == "parquet":
                    # Use Polars lazy scanning to efficiently count rows and features
                    lazy_df = pl.scan_parquet(file_path)
                    n_rows = lazy_df.select(pl.len()).collect().item()
                    n_cols = (
                        lazy_df.select(pl.col("inputCol").n_unique()).collect().item()
                    )

                    if n_cols > 0:
                        n_samples = n_rows // n_cols
                        position_rows = (
                            lazy_df.group_by(["sequenceId", "subsequenceId"])
                            .agg(
                                pl.col("leftPadLength").first(),
                                pl.col("startItemPosition").first(),
                                pl.col("splitStartItemPosition").first(),
                            )
                            .collect()
                        )
                        left_pad_lengths = position_rows.get_column(
                            "leftPadLength"
                        ).to_list()
                        target_valid_from = (
                            position_rows.select(
                                pl.max_horizontal(
                                    "leftPadLength",
                                    pl.col("splitStartItemPosition")
                                    - pl.col("startItemPosition"),
                                ).alias("targetValidFrom")
                            )
                            .get_column("targetValidFrom")
                            .to_list()
                        )
                        batch_files_metadata.append(
                            {
                                "path": file_path.name,
                                "samples": n_samples,
                                "left_pad_length_histogram": {
                                    str(value): count
                                    for value, count in Counter(
                                        left_pad_lengths
                                    ).items()
                                },
                                "target_valid_from_histogram": {
                                    str(value): count
                                    for value, count in Counter(
                                        target_valid_from
                                    ).items()
                                },
                            }
                        )
                        total_samples += n_samples
            except Exception as e:
                raise ValueError(
                    f"Could not validate {file_path} for metadata: {e}"
                ) from e

        metadata = {
            "split_index": split_index,
            "n_classes": getattr(self, "output_n_classes", {}),
            "column_data_types": getattr(self, "output_column_data_types", {}),
            "total_samples": total_samples,
            "batch_files": batch_files_metadata,
            **self._layout_metadata(),
        }
        if getattr(self, "has_sample_positions", False):
            metadata["curriculum_column"] = self.curriculum_column

        metadata_path = directory / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)


@beartype
def _reserved_input_columns(mask_column: Optional[str]) -> tuple[str, ...]:
    columns = INPUT_METADATA_COLUMNS
    if mask_column is None:
        return columns
    return (*columns, mask_column)


@beartype
def _get_data_columns(
    data: pl.DataFrame, mask_column: Optional[str] = None
) -> list[str]:
    return [
        col
        for col in data.columns
        if col not in _reserved_input_columns(mask_column)
        and not col.startswith(CURRICULUM_COLUMN_PREFIX)
        and col != SPLIT_VALUE_COLUMN
    ]


@beartype
def _deduplicate_columns(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


@beartype
def _selected_columns_with_optional_mask(
    data_path: str,
    read_format: str,
    selected_columns: Optional[list[str]],
    mask_column: Optional[str] = None,
    curriculum_column: Optional[Union[str, list[str]]] = None,
    split_column: Optional[str] = None,
) -> Optional[list[str]]:
    if selected_columns is None:
        return selected_columns
    schema_columns = None
    if os.path.exists(data_path):
        schema_columns = (
            pq.read_schema(data_path).names
            if read_format == "parquet"
            else pl.scan_csv(data_path).collect_schema().names()
        )
    optional_columns = [mask_column] if mask_column is not None else []
    optional_columns.extend(curriculum_columns(curriculum_column))
    if split_column is not None:
        optional_columns.append(split_column)
    if schema_columns is not None:
        if mask_column is not None and mask_column not in schema_columns:
            raise ValueError(f"mask_column '{mask_column}' not found in {data_path}")
    return _deduplicate_columns(selected_columns + optional_columns)


@beartype
def _validate_and_create_mask_column_expr(
    series: Any, mask_dtype: Any, mask_column: str
) -> pl.Expr:
    mask_col = pl.col(mask_column)

    if mask_dtype == pl.Boolean:
        return mask_col

    if mask_dtype.is_numeric():
        if not series.drop_nulls().is_in([0, 1]).all():
            raise ValueError(
                f"Mask column {mask_column} contains inadmissible values not in (0, 1)"
            )
        return mask_col == 1

    raise ValueError(
        f"Column {mask_column} must be boolean or numeric, got {mask_dtype}"
    )


@beartype
def _apply_mask_column(
    data: pl.DataFrame,
    data_columns: list[str],
    col_types: dict[str, str],
    mask_column: Optional[str] = None,
) -> pl.DataFrame:
    if mask_column is None:
        return data
    if mask_column not in data.columns:
        raise ValueError(f"mask_column '{mask_column}' not found in input data")

    mask_expr = _validate_and_create_mask_column_expr(
        data[mask_column], data.schema[mask_column], mask_column
    )
    updates = []
    for col in data_columns:
        mask_value = (
            SPECIAL_TOKEN_IDS.mask
            if is_integer_dtype_name(col_types[col])
            else REAL_MASK_VALUE
        )
        updates.append(
            pl.when(mask_expr)
            .then(pl.lit(mask_value))
            .otherwise(pl.col(col))
            .cast(data.schema[col])
            .alias(col)
        )

    if updates:
        data = data.with_columns(updates)

    return data.drop(mask_column)


def _is_hash_bucket_label(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(HASH_BUCKET_LABEL_PREFIX)
        and value.endswith("]")
    )


@beartype
def _cardinality_expression(
    data: pl.DataFrame,
    column: str,
    id_map: dict[Union[str, int], int],
    config: Mapping[str, Any],
) -> pl.Expr:
    """Encode retained values directly and route the remainder as configured."""
    source = pl.col(column)
    string_categorical = isinstance(
        data.schema[column], (pl.String, pl.Utf8, pl.Categorical)
    )
    hashing = config.get("hashing")
    direct_map: dict[Any, int] = {
        key: value
        for key, value in id_map.items()
        if key not in SPECIAL_TOKEN_LABELS
        and (hashing is None or not _is_hash_bucket_label(key))
    }
    # JSON object keys are strings, so restore retained numeric categories using
    # the actual input schema before building the Polars replacement expression.
    if data.schema[column] == pl.Boolean:
        direct_map = {bool(int(key)): value for key, value in direct_map.items()}
    elif data.schema[column].is_integer():
        direct_map = {int(key): value for key, value in direct_map.items()}

    if hashing is None:
        fallback = pl.lit(SPECIAL_TOKEN_IDS.other)
    else:
        bucket_ids = [
            value for key, value in id_map.items() if _is_hash_bucket_label(key)
        ]
        if not bucket_ids:
            raise ValueError(f"No hashing buckets found in ID map for {column!r}")
        seed = int(hashing.get("seed", 0))
        stable_hashes = source.map_elements(
            lambda value: _stable_category_hash(value, seed),
            return_dtype=pl.UInt64,
        )
        fallback = (
            pl.lit(min(bucket_ids), dtype=pl.UInt64)
            + stable_hashes % pl.lit(int(hashing["num_buckets"]), dtype=pl.UInt64)
        ).cast(pl.Int64)

    expression = pl.when(source.is_null()).then(pl.lit(SPECIAL_TOKEN_IDS.unknown))
    if string_categorical:
        expression = (
            expression.when(
                source == SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.unknown]
            )
            .then(pl.lit(SPECIAL_TOKEN_IDS.unknown))
            .when(source == SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.other])
            .then(pl.lit(SPECIAL_TOKEN_IDS.other))
        )
    if direct_map:
        expression = expression.when(source.is_in(list(direct_map))).then(
            source.replace_strict(
                direct_map,
                default=SPECIAL_TOKEN_IDS.other,
            )
        )
    return expression.otherwise(fallback).alias(column)


@beartype
def _apply_column_statistics(
    data: pl.DataFrame,
    data_columns: list[str],
    id_maps: dict[str, dict[Union[str, int], int]],
    selected_columns_statistics: dict[str, dict[str, float]],
    normalize_real_columns: bool,
    n_classes: Optional[dict[str, int]] = None,
    col_types: Optional[dict[str, str]] = None,
    cardinality_config: Optional[Mapping[str, dict[str, Any]]] = None,
) -> tuple[pl.DataFrame, dict[str, int], dict[str, str]]:
    """Apply categorical maps and optional numeric standardization."""
    _validate_cardinality_reserved_values(data, cardinality_config or {})
    col_types_was_provided = col_types is not None

    if n_classes is None:
        n_classes = {
            col: max(SPECIAL_TOKEN_IDS.user_start, max(id_maps[col].values()) + 1)
            for col in id_maps
        }

    if col_types is None:
        col_types = {col: str(data.schema[col]) for col in data_columns}

    missing_columns = [
        col
        for col in data_columns
        if col not in id_maps
        and col not in selected_columns_statistics
        and not _is_temporal_dtype_name(col_types.get(col, ""))
    ]
    if missing_columns:
        raise ValueError(
            "No unmasked examples found for columns: "
            f"{missing_columns}. Check the mask column or provide precomputed metadata."
        )

    expressions = []
    for col in data_columns:
        if col in id_maps:
            cardinality = (cardinality_config or {}).get(col, {})
            if cardinality:
                expressions.append(
                    _cardinality_expression(data, col, id_maps[col], cardinality)
                )
            else:
                expressions.append(
                    pl.col(col)
                    .replace_strict(id_maps[col], default=SPECIAL_TOKEN_IDS.other)
                    .alias(col)
                )
            if not col_types_was_provided:
                col_types[col] = "Int64"
        elif col in selected_columns_statistics:
            if not col_types_was_provided and not is_float_dtype_name(col_types[col]):
                col_types[col] = "Float64"
            if normalize_real_columns:
                expressions.append(
                    (
                        (pl.col(col) - selected_columns_statistics[col]["mean"])
                        / (selected_columns_statistics[col]["std"] + 1e-9)
                    ).alias(col)
                )

    if expressions:
        data = data.with_columns(expressions)

    return (data, n_classes, col_types)


@beartype
def load_precomputed_id_maps(
    project_root: str,
    data_columns: Optional[list[str]],
    required_maps: Optional[list[str]] = None,
) -> dict[str, dict[Union[str, int], int]]:
    """Load and validate precomputed ID maps."""
    custom_maps = {}
    path = os.path.join(project_root, "configs", "id_maps")

    if required_maps and not os.path.exists(path):
        raise FileNotFoundError(
            f"use_precomputed_maps specified {required_maps}, but 'configs/id_maps' folder does not exist."
        )

    if os.path.exists(path):
        for file in os.listdir(path):
            if file.endswith(".json"):
                col_name = os.path.splitext(file)[0]
                if data_columns is not None and col_name not in data_columns:
                    raise ValueError(
                        f"{file} does not correspond to any column in the data"
                    )

                with open(os.path.join(path, file), "r") as f:
                    m = {k: int(v) for k, v in json.load(f).items()}

                    if not len(m) > 0:
                        raise ValueError(f"map in {file} does not contain any values")
                    for (
                        reserved_key,
                        expected_value,
                    ) in SPECIAL_TOKEN_IDS.ids_by_label.items():
                        if reserved_key in m and m[reserved_key] != expected_value:
                            raise ValueError(
                                f"{reserved_key} in map {file} must map to {expected_value}"
                            )

                    user_values = [
                        value
                        for key, value in m.items()
                        if key not in SPECIAL_TOKEN_LABELS
                    ]
                    if not user_values:
                        raise ValueError(
                            f"map in {file} does not contain any non-reserved values"
                        )

                    min_val = min(user_values)
                    if min_val == 2:
                        raise ValueError(
                            f"Precomputed map {file} uses legacy user IDs starting at 2"
                        )
                    if min_val != SPECIAL_TOKEN_IDS.user_start:
                        raise ValueError(
                            f"minimum non-reserved value in map {file} is {min_val}, must be {SPECIAL_TOKEN_IDS.user_start}."
                        )
                    if any(value in SPECIAL_TOKEN_ID_VALUES for value in user_values):
                        raise ValueError(
                            f"non-reserved values in map {file} must not use reserved IDs {SPECIAL_TOKEN_ID_VALUES}"
                        )
                    if len(set(m.values())) != len(m.values()):
                        raise ValueError(f"map in {file} contains duplicate IDs")
                    custom_maps[col_name] = m
    if required_maps:
        missing_maps = [col for col in required_maps if col not in custom_maps]
        if missing_maps:
            raise ValueError(
                f"Missing precomputed maps for required columns: {missing_maps}. "
                f"Please ensure {missing_maps[0]}.json exists in configs/id_maps/"
            )

    return custom_maps


@beartype
def _get_column_statistics(
    data: pl.DataFrame,
    data_columns: list[str],
    id_maps: dict[str, dict[Union[str, int], int]],
    selected_columns_statistics: dict[str, dict[str, float]],
    n_rows_running_count: int,
    precomputed_id_maps: dict[str, dict[Union[str, int], int]],
    mask_column: Optional[str] = None,
    categorical_value_sets: Optional[dict[str, set[Any]]] = None,
    categorical_columns: Optional[list[str]] = None,
    real_columns: Optional[list[str]] = None,
    categorical_value_counts: Optional[dict[str, Counter]] = None,
    cardinality_config: Optional[Mapping[str, dict[str, Any]]] = None,
) -> tuple[
    dict[str, dict[Union[str, int], int]],
    dict[str, dict[str, float]],
]:
    """Update ID maps and numeric statistics from one chunk."""
    if mask_column is not None and mask_column in data.columns:
        mask_expr = _validate_and_create_mask_column_expr(
            data[mask_column], data.schema[mask_column], mask_column
        )
        data = data.filter(~mask_expr)

    if data.is_empty():
        return id_maps, selected_columns_statistics

    categorical_set = set(categorical_columns or [])
    real_set = set(real_columns or [])
    for data_col in data_columns:
        dtype = data.schema[data_col]
        inferred_categorical = isinstance(
            dtype, (pl.String, pl.Utf8, pl.Object, pl.Categorical, pl.Boolean)
        ) or isinstance(
            dtype,
            (
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
            ),
        )
        is_categorical = data_col in categorical_set or (
            data_col not in real_set and inferred_categorical
        )
        is_real = data_col in real_set or (
            data_col not in categorical_set
            and isinstance(dtype, (pl.Float16, pl.Float32, pl.Float64))
        )

        if data_col in (cardinality_config or {}) and not is_categorical:
            raise ValueError(
                f"cardinality_config column {data_col!r} is not categorical"
            )

        if is_categorical:
            if not inferred_categorical:
                raise ValueError(
                    f"Categorical column {data_col!r} has unsupported input dtype "
                    f"{dtype}; expected a string, boolean, categorical, or integer dtype."
                )
            if data_col not in precomputed_id_maps:
                cardinality = (cardinality_config or {}).get(data_col)
                if cardinality is not None:
                    if "min_freq" in cardinality or "top_k" in cardinality:
                        if categorical_value_counts is None:
                            raise ValueError(
                                "categorical_value_counts is required when fitting "
                                "frequency-based cardinality limits"
                            )
                        values = data.get_column(data_col).drop_nulls().to_list()
                        categorical_value_counts.setdefault(data_col, Counter()).update(
                            values
                        )
                elif categorical_value_sets is not None:
                    categorical_value_sets.setdefault(data_col, set()).update(
                        data.get_column(data_col).unique().to_list()
                    )
                else:
                    new_id_map = create_id_map(data, column=data_col)
                    id_maps[data_col] = combine_maps(
                        new_id_map, id_maps.get(data_col, {})
                    )
            else:
                logger.info(f"Applying precomputed map for {data_col}")
        elif is_real:
            if not _dtype_is_numeric(dtype):
                raise ValueError(
                    f"Real column {data_col!r} has non-numeric input dtype {dtype}."
                )
            if data_col in precomputed_id_maps:
                raise ValueError(
                    f"Column {data_col} is not categorical, precomputed map is invalid."
                )

            chunk_mean, chunk_std = _finite_mean_and_std(
                data.get_column(data_col), data_col
            )
            previous_stats = selected_columns_statistics.get(data_col)

            if previous_stats is None:
                combined_mean, combined_std = chunk_mean, chunk_std
                combined_count = data.shape[0]
            else:
                previous_count = int(previous_stats.get("count", n_rows_running_count))
                combined_mean, combined_std = get_combined_statistics(
                    data.shape[0],
                    chunk_mean,
                    chunk_std,
                    previous_count,
                    previous_stats["mean"],
                    previous_stats["std"],
                )
                combined_count = previous_count + data.shape[0]

            selected_columns_statistics[data_col] = {
                "std": combined_std,
                "mean": combined_mean,
                "count": float(combined_count),
            }
        elif isinstance(dtype, (pl.Date, pl.Datetime, pl.Time, pl.Duration)):
            continue
        else:
            raise ValueError(f"Column {data_col} has unsupported dtype: {dtype}")

    return id_maps, selected_columns_statistics


@beartype
def _finite_mean_and_std(column: pl.Series, column_name: str) -> tuple[float, float]:
    """Compute sample statistics without overflowing intermediate squares."""
    values = column.to_numpy().astype(np.float64, copy=False)
    if values.size == 0:
        raise ValueError(f"Cannot compute statistics for empty column {column_name!r}.")
    if not np.isfinite(values).all():
        raise ValueError(f"Column {column_name!r} contains non-finite values.")

    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0, 0.0

    scaled_values = values / scale
    mean = float(np.mean(scaled_values) * scale)
    std = 0.0 if values.size <= 1 else float(np.std(scaled_values, ddof=1) * scale)
    if not math.isfinite(mean) or not math.isfinite(std):
        raise ValueError(
            f"Column {column_name!r} has a numeric range too large for finite "
            "normalization statistics."
        )
    return mean, std


@beartype
def _validate_sequence_coordinates(
    data: pl.DataFrame, source: str, sort_output: bool = True
) -> pl.DataFrame:
    """Sort once and reject ambiguous or unrepresentable coordinates."""
    item_position_dtype = data.schema["itemPosition"]
    if not isinstance(
        item_position_dtype,
        (
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ),
    ):
        raise ValueError(
            f"itemPosition must have an integer dtype in {source}; "
            f"found {item_position_dtype}."
        )

    item_positions = data.get_column("itemPosition")
    if item_positions.null_count() > 0:
        raise ValueError(f"itemPosition contains null values in {source}.")
    if not item_positions.is_empty():
        min_position = int(item_positions.min())
        max_position = int(item_positions.max())
        if min_position < INT64_INFO.min or max_position > INT64_INFO.max:
            raise ValueError(
                f"itemPosition must fit in signed Int64 in {source}; found range "
                f"[{min_position}, {max_position}]."
            )

    coordinates = data.select(list(INPUT_METADATA_COLUMNS))
    raw_sequence_ids = coordinates.get_column("sequenceId").to_numpy()
    raw_positions = (
        coordinates.get_column("itemPosition").to_numpy().astype(np.int64, copy=False)
    )
    already_sorted = len(raw_sequence_ids) < 2 or not np.any(
        (raw_sequence_ids[1:] < raw_sequence_ids[:-1])
        | (
            (raw_sequence_ids[1:] == raw_sequence_ids[:-1])
            & (raw_positions[1:] < raw_positions[:-1])
        )
    )
    if already_sorted:
        ordered = data if sort_output else coordinates
    else:
        ordered = (
            data.sort(list(INPUT_METADATA_COLUMNS))
            if sort_output
            else coordinates.sort(list(INPUT_METADATA_COLUMNS))
        )
    sequence_ids = ordered.get_column("sequenceId").to_numpy()
    positions = (
        ordered.get_column("itemPosition").to_numpy().astype(np.int64, copy=False)
    )
    if len(sequence_ids) > 1:
        same_sequence = sequence_ids[1:] == sequence_ids[:-1]
        position_differences = positions[1:] - positions[:-1]
        invalid_indices = np.flatnonzero(same_sequence & (position_differences != 1))
    else:
        invalid_indices = np.array([], dtype=np.int64)

    if len(invalid_indices) > 0:
        row_index = int(invalid_indices[0]) + 1
        sample = ordered.select(list(INPUT_METADATA_COLUMNS)).row(row_index)
        if positions[row_index] == positions[row_index - 1]:
            raise ValueError(
                "duplicate (sequenceId, itemPosition) coordinate found in "
                f"{source}: {sample}"
            )
        raise ValueError(
            "itemPosition must increase by one within each sequence because the "
            f"stored window format cannot represent gaps; found {sample} in {source}."
        )

    return ordered if sort_output else data


@beartype
def _load_and_preprocess_data(
    data_path: str,
    read_format: str,
    selected_columns: Optional[list[str]],
    max_rows: Optional[int],
    mask_column: Optional[str] = None,
    curriculum_column: Optional[Union[str, list[str]]] = None,
    sort_rows: bool = True,
    split_column: Optional[str] = None,
    split_values: Optional[list[Any]] = None,
) -> pl.DataFrame:
    """Read, validate, column-filter, and row-limit one input file."""
    logger.info(f"Reading data from '{data_path}'...")
    columns_to_read = _selected_columns_with_optional_mask(
        data_path,
        read_format,
        selected_columns,
        mask_column,
        curriculum_column,
        split_column,
    )
    data = read_data(data_path, read_format, columns=columns_to_read)
    if SPLIT_VALUE_COLUMN in data.columns:
        raise ValueError(
            f"Input column name {SPLIT_VALUE_COLUMN!r} is reserved for "
            "value-cutoff preprocessing"
        )

    configured_curriculum_columns = curriculum_columns(curriculum_column)
    for column in configured_curriculum_columns:
        if column not in data.columns:
            raise ValueError(f"curriculum_column '{column}' not found in {data_path}")
    data = data.rename(
        {
            column: curriculum_storage_column(column)
            for column in configured_curriculum_columns
        }
    )

    if mask_column is not None and mask_column not in data.columns:
        raise ValueError(f"mask_column '{mask_column}' not found in {data_path}")

    if data.null_count().sum().sum_horizontal().item() != 0:
        raise ValueError(f"NaN or null values not accepted: {data.null_count()}")

    if selected_columns:
        selected_columns_filtered = [
            (
                curriculum_storage_column(col)
                if col in configured_curriculum_columns
                else col
            )
            for col in selected_columns
            if col not in INPUT_METADATA_COLUMNS
        ]
        columns_to_select = list(INPUT_METADATA_COLUMNS) + selected_columns_filtered
        if mask_column is not None and mask_column in data.columns:
            columns_to_select.append(mask_column)
        columns_to_select.extend(curriculum_storage_columns(data.columns))
        if split_column is not None:
            columns_to_select.append(split_column)
        data = data.select(_deduplicate_columns(columns_to_select))

    if max_rows:
        data = data.slice(0, int(max_rows))

    output_split_column = selected_columns is None or split_column in selected_columns
    data = _add_normalized_split_column(data, split_column, split_values, data_path)
    unsupported_temporal_columns = [
        column
        for column, dtype in data.schema.items()
        if isinstance(dtype, (pl.Date, pl.Datetime, pl.Time, pl.Duration))
        and column != split_column
    ]
    if unsupported_temporal_columns:
        raise ValueError(
            "Timestamp columns are only supported as the value_cutoff "
            f"split_column; found {unsupported_temporal_columns} in {data_path}"
        )
    if split_column is not None and not output_split_column:
        data = data.drop(split_column)

    float_columns = [
        column
        for column, dtype in data.schema.items()
        if isinstance(dtype, (pl.Float16, pl.Float32, pl.Float64))
    ]
    non_finite_counts = (
        data.select(
            [
                pl.col(column).is_finite().not_().sum().alias(column)
                for column in float_columns
            ]
        ).row(0, named=True)
        if float_columns
        else {}
    )
    non_finite_counts = {
        column: count for column, count in non_finite_counts.items() if count > 0
    }
    if non_finite_counts:
        raise ValueError(
            f"non-finite real values are not accepted in {data_path}: "
            f"{non_finite_counts}"
        )
    for column in curriculum_storage_columns(data.columns):
        position_dtype = data.schema[column]
        if not position_dtype.is_integer():
            raise ValueError(
                f"{column} must have an integer dtype in {data_path}; "
                f"found {position_dtype}."
            )
        data = data.with_columns(pl.col(column).cast(pl.Int64))

    try:
        data = _validate_sequence_coordinates(data, data_path, sort_rows)
    except ValueError as error:
        if "duplicate" in str(error):
            source_schema = (
                pl.scan_csv(data_path)
                if read_format == "csv"
                else pl.scan_parquet(data_path)
            ).collect_schema()
            if "subItemPosition" in source_schema:
                raise ValueError(
                    f"{error}. For repeated child rows, explicitly configure depth_layouts with position_column: subItemPosition"
                ) from error
        raise

    return data


@beartype
def _check_file_has_been_processed(
    project_root: str,
    data_name_root: str,
    process_id: int,
    n_splits: int,
    write_format: str,
    target_dir: str,
    merge_output: bool,
    file_index_str: str,
):
    file_prefix_str = f"{data_name_root}-{process_id}-{file_index_str}"

    if merge_output:
        # Case 1: Combining into a single file. Check for the intermediate
        # combined file in the target_dir.
        expected_file_path = ""
        for split_index in range(n_splits):
            expected_file_path = create_split_file_path(
                project_root,
                data_name_root,
                split_index,
                write_format,
                in_target_dir=True,  # Intermediate files are in target_dir
                target_dir=target_dir,
                pre_split_str=file_prefix_str,  # This file's unique ID
                post_split_str=None,
            )
            if not os.path.exists(expected_file_path):
                # If any split's intermediate file is missing, we must re-process
                return False
        logger.info(
            f"Files: {expected_file_path.split('split')[0] + 'splitX'} found, skipping"
        )
        return True
    else:
        temp_dir_path = os.path.join(project_root, "data", target_dir)

        if not os.path.isdir(temp_dir_path):
            return False

        for file_name in os.listdir(temp_dir_path):
            if file_name.startswith(file_prefix_str) and file_name.endswith(
                f".{write_format}"
            ):
                logger.info(f"Found {file_name}, skipping corresponding input file...")
                return True

        return False


@beartype
def _get_processed_prefixes(
    project_root: str,
    target_dir: str,
    write_format: str,
) -> set[str]:
    temp_dir = Path(project_root) / "data" / target_dir

    if not temp_dir.is_dir():
        return set()

    suffix = f".{write_format}"
    processed = set()

    with os.scandir(temp_dir) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith(suffix):
                continue

            if "-split" in entry.name:
                processed.add(entry.name.rsplit("-split", 1)[0])

    return processed


@beartype
def _process_batches_multiple_files_inner(
    project_root: str,
    data_name_root: str,
    process_id: int,
    file_paths: list[str],
    read_format: str,
    selected_columns: Optional[list[str]],
    max_rows: Optional[int],
    schema: Any,
    n_cores: int,
    layout: StoredWindowLayout,
    window_stride: int,
    data_columns: list[str],
    n_classes: dict[str, int],
    id_maps: dict[str, dict[Union[int, str], int]],
    selected_columns_statistics: dict[str, dict[str, float]],
    col_types: dict[str, str],
    split_ratios: Optional[list[float]],
    write_format: str,
    split_paths: list[str],
    target_dir: str,
    batches_per_file: int,
    merge_output: bool,
    allow_sequence_splitting: bool,
    continue_preprocessing: bool,
    mask_column: Optional[str],
    split_method: str,
    seed: int,
    normalize_real_columns: bool,
    sequence_split_assignments: Optional[dict[int, int]],
    curriculum_column: Optional[Union[str, list[str]]],
    split_column: Optional[str],
    split_values: Optional[list[Any]],
    normalize_on_all_data: bool = True,
    alignment: Optional[PredictionAlignment] = None,
    cardinality_config: Optional[dict[str, dict[str, Any]]] = None,
):
    """Process this worker's file shard."""

    n_splits = _split_count(split_ratios, split_values)

    processed_prefixes = (
        _get_processed_prefixes(project_root, target_dir, write_format)
        if continue_preprocessing and not merge_output
        else set()
    )
    n_files = len(file_paths)
    if n_files <= 0:
        raise ValueError("No files found to process.")
    pad_width = len(str(n_files - 1))
    n_rows_running_count = 0
    worker_pool = _create_preprocess_pool(n_cores) if n_cores > 1 else None
    for file_index, path in enumerate(file_paths):
        max_rows_inner = None if max_rows is None else max_rows - n_rows_running_count
        if max_rows_inner is None or max_rows_inner > 0:
            file_index_str = str(file_index).zfill(pad_width)

            adjusted_split_paths = [
                path.replace(
                    data_name_root, f"{data_name_root}-{process_id}-{file_index_str}"
                )
                for path in split_paths
            ]
            if continue_preprocessing:
                file_prefix_str = f"{data_name_root}-{process_id}-{file_index_str}"

                if not merge_output:
                    file_has_been_processed = file_prefix_str in processed_prefixes
                else:
                    file_has_been_processed = _check_file_has_been_processed(
                        project_root,
                        data_name_root,
                        process_id,
                        n_splits,
                        write_format,
                        target_dir,
                        merge_output,
                        file_index_str,
                    )
                if file_has_been_processed:
                    logger.info(f"Skipping already processed file: {path}")
                    if max_rows is not None:
                        data = _load_and_preprocess_data(
                            path,
                            read_format,
                            selected_columns,
                            max_rows_inner,
                            mask_column,
                            curriculum_column,
                            split_column=split_column,
                            split_values=split_values,
                        )
                        n_rows_running_count += data.shape[0]
                    continue

            data = _load_and_preprocess_data(
                path,
                read_format,
                selected_columns,
                max_rows_inner,
                mask_column,
                curriculum_column,
                split_column=split_column,
                split_values=split_values,
            )
            data = _apply_configured_input_casting(data, data_columns, col_types)
            data, _, _ = _apply_column_statistics(
                data,
                data_columns,
                id_maps,
                selected_columns_statistics,
                normalize_real_columns,
                n_classes,
                col_types,
                cardinality_config,
            )
            data = _apply_mask_column(data, data_columns, col_types, mask_column)
            data = _apply_output_type_casting(data, data_columns, col_types)

            data_name_root_inner = f"{data_name_root}-{process_id}-{file_index_str}"

            n_batches = _process_batches_single_file(
                project_root,
                data_name_root_inner,
                data,
                schema,
                n_cores,
                layout,
                window_stride,
                data_columns,
                col_types,
                split_ratios,
                write_format,
                adjusted_split_paths,
                target_dir,
                batches_per_file,
                merge_output,
                allow_sequence_splitting,
                split_method,
                seed,
                sequence_split_assignments,
                worker_pool,
                normalize_on_all_data=normalize_on_all_data,
                alignment=alignment,
                split_values=split_values,
            )

            if merge_output:
                input_files = create_file_paths_for_multiple_files1(
                    project_root,
                    target_dir,
                    n_splits,
                    n_batches,
                    process_id,
                    file_index_str,
                    data_name_root,
                    write_format,
                )
                combine_multiprocessing_outputs(
                    project_root,
                    target_dir,
                    n_splits,
                    input_files,
                    data_name_root,
                    write_format,
                    in_target_dir=True,
                    pre_split_str=f"{process_id}-{file_index_str}",
                )

                delete_files(input_files)

            n_rows_running_count += data.shape[0]

    if worker_pool is not None:
        worker_pool.close()
        worker_pool.join()


@beartype
def _process_batches_single_file(
    project_root: str,
    data_name_root: str,
    data: pl.DataFrame,
    schema: Any,
    n_cores: Optional[int],
    layout: StoredWindowLayout,
    window_stride: int,
    data_columns: list[str],
    col_types: dict[str, str],
    split_ratios: Optional[list[float]],
    write_format: str,
    split_paths: list[str],
    target_dir: str,
    batches_per_file: int,
    merge_output: bool,
    allow_sequence_splitting: bool,
    split_method: str = "within_sequence",
    seed: int = 1010,
    sequence_split_assignments: Optional[dict[int, int]] = None,
    worker_pool: Optional[Any] = None,
    normalize_on_all_data: bool = True,
    alignment: Optional[PredictionAlignment] = None,
    split_values: Optional[list[Any]] = None,
) -> int:
    """Split one file into worker batches and preprocess them."""
    _split_count(split_ratios, split_values)
    if split_method == "value_cutoff" and data.height > 1:
        sequence_ids = data.get_column("sequenceId").to_numpy()
        values = data.get_column(SPLIT_VALUE_COLUMN).to_numpy()
        if np.any((sequence_ids[1:] == sequence_ids[:-1]) & (values[1:] < values[:-1])):
            raise ValueError(
                "split_column values must be non-decreasing within each sequence"
            )
    n_cores = n_cores or multiprocessing.cpu_count()
    sequence_count = len(_sequence_run_bounds(data)[0])
    maximum_batches = data.height if allow_sequence_splitting else sequence_count
    n_cores = min(n_cores, maximum_batches)
    if (
        not normalize_on_all_data
        and allow_sequence_splitting
        and split_method == "within_sequence"
        and n_cores > sequence_count
    ):
        raise ValueError(
            "Unsafe preprocessing configuration: normalize_on_all_data=False, "
            "allow_sequence_splitting=True, and split_method='within_sequence' "
            "would split sequences across worker batches "
            f"(effective n_cores={n_cores}, sequences={sequence_count}). This "
            "would make split-0 metadata fitting disagree with the materialized "
            "training split. Set allow_sequence_splitting=False, reduce n_cores "
            f"to at most {sequence_count}, or set normalize_on_all_data=True."
        )
    batch_limits = get_batch_limits(data, n_cores, allow_sequence_splitting)
    valid_batch_limits = [(s, e) for s, e in batch_limits if (e - s) > 0]
    batches = [
        (
            project_root,
            data_name_root,
            process_id,
            data.slice(start, end - start),
            schema,
            split_paths,
            layout,
            window_stride,
            data_columns,
            col_types,
            split_ratios,
            target_dir,
            write_format,
            batches_per_file,
            merge_output,
            split_method,
            seed,
            sequence_split_assignments,
            alignment,
            split_values,
        )
        for process_id, (start, end) in enumerate(valid_batch_limits)
    ]

    if len(batches) > 1:
        if worker_pool is not None:
            worker_pool.starmap(preprocess_batch, batches)
        else:
            with _create_preprocess_pool(len(batches)) as pool:
                pool.starmap(preprocess_batch, batches)
    else:
        preprocess_batch(*batches[0])

    return len(batches)


def _initialize_preprocess_worker() -> None:
    """Keep native thread pools from oversubscribing multiprocessing workers."""
    os.environ["POLARS_MAX_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _create_preprocess_pool(processes: int) -> Any:
    """Spawn workers with native compute pools limited before module import."""
    thread_variables = {
        "POLARS_MAX_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    previous_values = {name: os.environ.get(name) for name in thread_variables}
    os.environ.update(thread_variables)
    try:
        return multiprocessing.get_context("spawn").Pool(
            processes=processes,
            initializer=_initialize_preprocess_worker,
        )
    finally:
        for name, value in previous_values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@beartype
def get_combined_statistics(
    n1: int, mean1: float, std1: float, n2: int, mean2: float, std2: float
) -> tuple[float, float]:
    """Combine two mean/std summaries."""
    if n1 == 0:
        return mean2, std2
    if n2 == 0:
        return mean1, std1

    scale = max(abs(mean1), abs(mean2), abs(std1), abs(std2))
    if not math.isfinite(scale):
        raise ValueError("Cannot combine non-finite normalization statistics.")
    if scale == 0.0:
        return 0.0, 0.0

    scaled_mean1 = mean1 / scale
    scaled_mean2 = mean2 / scale
    scaled_std1 = std1 / scale
    scaled_std2 = std2 / scale
    combined_mean_scaled = (n1 * scaled_mean1 + n2 * scaled_mean2) / (n1 + n2)
    combined_mean = combined_mean_scaled * scale

    if n1 + n2 <= 1:
        return combined_mean, 0.0

    sum_of_squares1 = (n1 - 1) * scaled_std1**2 + n1 * (
        scaled_mean1 - combined_mean_scaled
    ) ** 2
    sum_of_squares2 = (n2 - 1) * scaled_std2**2 + n2 * (
        scaled_mean2 - combined_mean_scaled
    ) ** 2

    combined_std = (
        math.sqrt((sum_of_squares1 + sum_of_squares2) / (n1 + n2 - 1)) * scale
    )
    if not math.isfinite(combined_mean) or not math.isfinite(combined_std):
        raise ValueError(
            "Combined numeric range is too large for finite normalization statistics."
        )

    return combined_mean, combined_std


@beartype
def create_id_map(data: pl.DataFrame, column: str) -> dict[Union[str, int], int]:
    """Map sorted user values to IDs after reserved tokens."""
    ids = sorted(
        [int(x) if not isinstance(x, str) else x for x in np.unique(data[column])]
    )  # type: ignore

    if isinstance(ids[0], str):
        if SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.mask] in ids:
            raise ValueError(
                f"Found value '{SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.mask]}' in {column}, this is invalid"
            )

        for special_val in [
            SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.unknown],
            SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.other],
        ]:
            if special_val in ids:
                warnings.warn(
                    f"Found special value {special_val} in {column}, these will be combined with the sequifier-internal special value {special_val}"
                )
        ids = [
            id_
            for id_ in ids
            if id_
            not in [
                SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.unknown],
                SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.other],
            ]
        ]
        id_map = {id_: i + SPECIAL_TOKEN_IDS.user_start for i, id_ in enumerate(ids)}
        id_map[SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.unknown]] = (
            SPECIAL_TOKEN_IDS.unknown
        )
        id_map[SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.other]] = (
            SPECIAL_TOKEN_IDS.other
        )
    else:
        id_map = {id_: i + SPECIAL_TOKEN_IDS.user_start for i, id_ in enumerate(ids)}
    return dict(id_map)


@beartype
def get_batch_limits(
    data: pl.DataFrame, n_batches: int, allow_sequence_splitting: bool
) -> list[tuple[int, int]]:
    """Split rows into batches without crossing sequenceId boundaries unless allowed."""
    if n_batches <= 0:
        raise ValueError("n_batches must be positive.")
    if data.is_empty():
        raise ValueError("Cannot split an empty dataset into batches.")

    sequence_ids = data.get_column("sequenceId").to_numpy()
    sequence_start_indices = np.concatenate(
        [[0], np.where(sequence_ids[1:] != sequence_ids[:-1])[0] + 1]
    )
    sequence_boundaries = np.concatenate([sequence_start_indices, [data.shape[0]]])
    sequence_count = len(sequence_start_indices)

    if n_batches > sequence_count:
        if not allow_sequence_splitting:
            raise ValueError(
                "Cannot create more non-empty batches than there are sequences without "
                "splitting a sequence."
            )

        original_lengths = np.diff(sequence_boundaries)
        pieces = np.ones(len(original_lengths), dtype=int)

        for _ in range(n_batches - sequence_count):
            largest_piece_idx = int(np.argmax(original_lengths / pieces))
            pieces[largest_piece_idx] += 1

        if np.any(pieces > original_lengths):
            raise ValueError(
                "Cannot split further: sequences are too short to reach the "
                "requested number of non-empty batches."
            )

        new_boundaries = []
        for start, length, num_pieces in zip(
            sequence_boundaries[:-1], original_lengths, pieces
        ):
            # Calculate evenly spaced boundaries within this sequence
            splits = start + np.round(np.linspace(0, length, num_pieces + 1)).astype(
                int
            )

            if not new_boundaries:
                new_boundaries.extend(splits)
            else:
                new_boundaries.extend(
                    splits[1:]
                )  # Avoid duplicating the shared boundary

        sequence_boundaries = np.array(new_boundaries)

    interior_boundaries = sequence_boundaries[1:-1]
    ideal_limits = np.linspace(0, data.shape[0], n_batches + 1)[1:-1]

    selected_boundaries: list[int] = []
    previous_boundary = 0
    for batch_index, ideal_limit in enumerate(ideal_limits):
        remaining_boundaries_needed = len(ideal_limits) - batch_index - 1
        candidates = [
            int(boundary)
            for boundary in interior_boundaries
            if boundary > previous_boundary
            and (data.shape[0] - boundary) >= remaining_boundaries_needed
        ]
        if not candidates:
            raise ValueError(
                "Cannot create requested non-empty batches without splitting a sequence."
            )

        selected_boundary = min(
            candidates,
            key=lambda boundary: abs(boundary - ideal_limit),
        )
        selected_boundaries.append(selected_boundary)
        previous_boundary = selected_boundary

    limits = [0, *selected_boundaries, data.shape[0]]
    return list(zip(limits[:-1], limits[1:]))


@beartype
def _sequence_run_bounds(batch: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return contiguous sequence run starts and stops for an ordered batch."""
    if batch.is_empty():
        empty = np.array([], dtype=np.int64)
        return empty, empty

    sequence_ids = batch.get_column("sequenceId").to_numpy()
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(sequence_ids[1:] != sequence_ids[:-1]).astype(np.int64) + 1,
        )
    )
    stops = np.concatenate((starts[1:], np.array([len(sequence_ids)], dtype=np.int64)))
    return starts, stops


def _batch_to_arrays(batch: pl.DataFrame, data_columns: list[str]) -> BatchArrays:
    """Convert a worker batch to NumPy once for all sequence extractions."""
    starts, stops = _sequence_run_bounds(batch)
    curriculum_value_columns = curriculum_storage_columns(batch.columns)
    return BatchArrays(
        sequence_ids=batch.get_column("sequenceId").to_numpy(),
        item_positions=batch.get_column("itemPosition").to_numpy(),
        values={column: batch.get_column(column).to_numpy() for column in data_columns},
        curriculum_values={
            column: batch.get_column(column).to_numpy()
            for column in curriculum_value_columns
        },
        split_values=(
            batch.get_column(SPLIT_VALUE_COLUMN).to_numpy()
            if SPLIT_VALUE_COLUMN in batch.columns
            else None
        ),
        run_starts=starts,
        run_stops=stops,
    )


@beartype
def _iter_sequence_runs(
    batch: pl.DataFrame,
    starts: np.ndarray,
    stops: np.ndarray,
) -> Iterator[tuple[int, pl.DataFrame]]:
    """Yield sequence IDs and zero-copy-friendly contiguous DataFrame slices."""
    sequence_ids = batch.get_column("sequenceId")
    for start, stop in zip(starts, stops):
        start_index = int(start)
        stop_index = int(stop)
        yield (
            int(sequence_ids[start_index]),
            batch.slice(start_index, stop_index - start_index),
        )


@beartype
def combine_maps(
    map1: dict[Union[str, int], int], map2: dict[Union[str, int], int]
) -> dict[Union[str, int], int]:
    """Merge maps and reassign user IDs after reserved tokens."""
    keys1 = {k for k in map1.keys() if k not in SPECIAL_TOKEN_LABELS}
    keys2 = {k for k in map2.keys() if k not in SPECIAL_TOKEN_LABELS}

    combined_keys = sorted(list(keys1.union(keys2)))
    id_map = {
        id_: i + SPECIAL_TOKEN_IDS.user_start for i, id_ in enumerate(combined_keys)
    }

    if combined_keys and isinstance(combined_keys[0], str):
        id_map[SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.unknown]] = (
            SPECIAL_TOKEN_IDS.unknown
        )
        id_map[SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.other]] = (
            SPECIAL_TOKEN_IDS.other
        )

    return id_map


@beartype
def get_group_bounds(data_subset: pl.DataFrame, split_ratios: list[float]):
    """Return per-split row bounds for one sequence."""
    n = data_subset.shape[0]
    upper_bounds = list((np.cumsum(split_ratios) * n).astype(int))
    lower_bounds = [0] + list(upper_bounds[:-1])
    group_bounds = list(zip(lower_bounds, upper_bounds))
    return group_bounds


@beartype
def _balanced_sequence_split_assignments(
    sequence_ids: list[int], split_ratios: list[float], seed: int
) -> dict[int, int]:
    """Return sparse overrides ensuring deterministic splits are non-empty."""
    unique_sequence_ids = sorted(set(sequence_ids))
    if len(unique_sequence_ids) < len(split_ratios):
        raise ValueError(
            "between_sequence splitting needs at least as many sequences as requested "
            f"splits; found {len(unique_sequence_ids)} sequences for "
            f"{len(split_ratios)} splits."
        )

    counts: Counter[int] = Counter()
    candidates: dict[int, list[tuple[int, int]]] = {
        split: [] for split in range(len(split_ratios))
    }
    candidate_limit = len(split_ratios)
    for sequence_id in unique_sequence_ids:
        split = assign_sequence_to_split(sequence_id, split_ratios, seed)
        counts[split] += 1
        digest_value = int.from_bytes(
            hashlib.sha256(f"{seed}:{sequence_id}".encode("utf-8")).digest(),
            byteorder="big",
            signed=False,
        )
        candidate = (digest_value, sequence_id)
        if len(candidates[split]) < candidate_limit:
            heapq.heappush(candidates[split], candidate)
        elif candidate > candidates[split][0]:
            heapq.heapreplace(candidates[split], candidate)

    missing_splits = [split for split in range(len(split_ratios)) if counts[split] == 0]
    overrides = {}
    for missing_split in missing_splits:
        donor_split = max(
            (split for split, count in counts.items() if count > 1),
            key=lambda split: (counts[split], split_ratios[split], -split),
        )
        _, sequence_to_move = heapq.nlargest(1, candidates[donor_split])[0]
        candidates[donor_split] = [
            candidate
            for candidate in candidates[donor_split]
            if candidate[1] != sequence_to_move
        ]
        heapq.heapify(candidates[donor_split])
        overrides[sequence_to_move] = missing_split
        counts[donor_split] -= 1
        counts[missing_split] += 1

    return overrides


@beartype
def process_and_write_data_pt(
    data: pl.DataFrame,
    window_length: int,
    path: str,
    column_data_types: dict[str, str],
):
    """Write long-format sequences as packed PT tensors."""
    if data.is_empty():
        return

    sequence_cols = [str(c) for c in range(window_length - 1, -1, -1)]

    all_feature_cols = data.get_column("inputCol").unique().to_list()
    curriculum_value_columns = curriculum_storage_columns(data.columns)
    split_bound_columns = {
        "splitStartItemPosition",
        "splitEndItemPosition",
    }
    missing_split_bound_columns = split_bound_columns - set(data.columns)
    if missing_split_bound_columns:
        raise ValueError(
            "Preprocessed windows require split target bounds; missing columns: "
            f"{sorted(missing_split_bound_columns)}"
        )

    aggs = [
        pl.concat_list(sequence_cols)
        .filter(pl.col("inputCol") == col_name)
        .list.explode(keep_nulls=False, empty_as_null=False)  # flatten
        .alias(f"seq_{col_name}")
        for col_name in all_feature_cols
    ] + [
        pl.col("startItemPosition").first().alias("startItemPosition"),
        pl.col("leftPadLength").first().alias("leftPadLength"),
    ]
    aggs.extend(
        [
            pl.col("splitStartItemPosition").first().alias("splitStartItemPosition"),
            pl.col("splitEndItemPosition").first().alias("splitEndItemPosition"),
        ]
    )
    for index, column in enumerate(curriculum_value_columns):
        aggs.extend(
            [
                pl.col(column).first().alias(column),
                pl.col(column).n_unique().alias(f"__sample_position_count_{index}"),
            ]
        )

    sort_columns = (
        [*curriculum_value_columns, "sequenceId", "subsequenceId"]
        if curriculum_value_columns
        else ["sequenceId", "subsequenceId"]
    )
    aggregated_data = data.group_by(["sequenceId", "subsequenceId"]).agg(aggs)
    if any(
        aggregated_data.get_column(f"__sample_position_count_{index}").max() != 1
        for index in range(len(curriculum_value_columns))
    ):
        raise ValueError("Curriculum columns must be identical per subsequence")
    aggregated_data = aggregated_data.sort(sort_columns)

    if aggregated_data.is_empty():
        return

    sequence_ids_tensor = torch.tensor(
        aggregated_data.get_column("sequenceId").to_numpy(), dtype=torch.int64
    )
    subsequence_ids_tensor = torch.tensor(
        aggregated_data.get_column("subsequenceId").to_numpy(), dtype=torch.int64
    )
    start_item_positions_tensor = torch.tensor(
        aggregated_data.get_column("startItemPosition").to_numpy(), dtype=torch.int64
    )
    left_pad_lengths_tensor = torch.tensor(
        aggregated_data.get_column("leftPadLength").to_numpy(), dtype=torch.int64
    )
    split_start_item_positions_tensor = torch.tensor(
        aggregated_data.get_column("splitStartItemPosition").to_numpy(),
        dtype=torch.int64,
    )
    split_end_item_positions_tensor = torch.tensor(
        aggregated_data.get_column("splitEndItemPosition").to_numpy(),
        dtype=torch.int64,
    )
    sample_positions_tensor = (
        torch.tensor(
            aggregated_data.select(curriculum_value_columns).to_numpy(),
            dtype=torch.int64,
        )
        if curriculum_value_columns
        else None
    )
    if sample_positions_tensor is not None and len(curriculum_value_columns) == 1:
        sample_positions_tensor = sample_positions_tensor[:, 0]
    sequences_dict = {}

    for col_name in all_feature_cols:
        torch_dtype = _torch_dtype_for_column(column_data_types[col_name])

        sequences_np = np.vstack(
            aggregated_data.get_column(f"seq_{col_name}").to_numpy(writable=True)
        )

        sequences_dict[col_name] = torch.tensor(sequences_np, dtype=torch_dtype)

    if not sequences_dict:
        return

    logger.info(f"Writing preprocessed data to '{path}'...")
    data_to_save = (
        sequences_dict,
        sequence_ids_tensor,
        subsequence_ids_tensor,
        start_item_positions_tensor,
        left_pad_lengths_tensor,
    )
    save_pt_payload(
        StoredTensorBatch(
            *data_to_save,
            split_start_item_positions=split_start_item_positions_tensor,
            split_end_item_positions=split_end_item_positions_tensor,
            sample_positions=sample_positions_tensor,
            curriculum_columns=tuple(
                source_curriculum_column(column) for column in curriculum_value_columns
            ),
        ),
        path,
    )


@beartype
def process_and_write_sequence_windows_pt(
    windows: list[SequenceWindows],
    path: str,
    column_data_types: dict[str, str],
) -> None:
    """Write dense sequence windows directly as a packed PT payload."""
    windows = [sequence for sequence in windows if sequence.n_samples > 0]
    if not windows:
        return

    feature_names = list(windows[0].values)
    curriculum_value_columns = windows[0].curriculum_columns
    for sequence in windows:
        if list(sequence.values) != feature_names:
            raise ValueError("All accumulated sequences must have the same features")
        if sequence.curriculum_columns != curriculum_value_columns:
            raise ValueError(
                "All accumulated sequences must have the same curriculum columns"
            )
        if bool(sequence.sample_positions is not None) != bool(
            curriculum_value_columns
        ):
            raise ValueError(
                "Sample positions must be present exactly when curriculum columns "
                "are configured"
            )

    sequence_ids = np.concatenate(
        [
            np.full(sequence.n_samples, sequence.sequence_id, dtype=np.int64)
            for sequence in windows
        ]
    )
    subsequence_ids = np.concatenate([sequence.subsequence_ids for sequence in windows])
    start_item_positions = np.concatenate(
        [sequence.start_item_positions for sequence in windows]
    )
    left_pad_lengths = np.concatenate(
        [sequence.left_pad_lengths for sequence in windows]
    )
    split_start_item_positions = np.concatenate(
        [sequence.split_start_item_positions for sequence in windows]
    )
    split_end_item_positions = np.concatenate(
        [sequence.split_end_item_positions for sequence in windows]
    )
    sample_positions = (
        np.concatenate(
            [
                sequence.sample_positions
                for sequence in windows
                if sequence.sample_positions is not None
            ]
        )
        if curriculum_value_columns
        else None
    )

    sort_order = None
    if sample_positions is not None:
        sort_keys = [
            sample_positions[:, index] for index in range(sample_positions.shape[1])
        ]
        sort_keys.extend((sequence_ids, subsequence_ids))
        candidate_order = np.lexsort(tuple(reversed(sort_keys)))
        if not np.array_equal(candidate_order, np.arange(len(candidate_order))):
            sort_order = candidate_order

    def ordered(values: np.ndarray) -> np.ndarray:
        return (
            values if sort_order is None else np.ascontiguousarray(values[sort_order])
        )

    sequence_ids_tensor = torch.from_numpy(ordered(sequence_ids))
    subsequence_ids_tensor = torch.from_numpy(ordered(subsequence_ids))
    start_item_positions_tensor = torch.from_numpy(ordered(start_item_positions))
    left_pad_lengths_tensor = torch.from_numpy(ordered(left_pad_lengths))
    split_start_item_positions_tensor = torch.from_numpy(
        ordered(split_start_item_positions)
    )
    split_end_item_positions_tensor = torch.from_numpy(
        ordered(split_end_item_positions)
    )
    sample_positions_tensor = (
        torch.from_numpy(ordered(sample_positions))
        if sample_positions is not None
        else None
    )
    if sample_positions_tensor is not None and len(curriculum_value_columns) == 1:
        sample_positions_tensor = sample_positions_tensor[:, 0]

    sequences_dict = {}
    for column in feature_names:
        values = np.concatenate([sequence.values[column] for sequence in windows])
        tensor = torch.from_numpy(ordered(values))
        target_dtype = _torch_dtype_for_column(column_data_types[column])
        sequences_dict[column] = (
            tensor if tensor.dtype == target_dtype else tensor.to(dtype=target_dtype)
        )

    logger.info(f"Writing preprocessed data to '{path}'...")
    save_pt_payload(
        StoredTensorBatch(
            sequences_dict,
            sequence_ids_tensor,
            subsequence_ids_tensor,
            start_item_positions_tensor,
            left_pad_lengths_tensor,
            split_start_item_positions=split_start_item_positions_tensor,
            split_end_item_positions=split_end_item_positions_tensor,
            sample_positions=sample_positions_tensor,
            curriculum_columns=tuple(
                source_curriculum_column(column) for column in curriculum_value_columns
            ),
        ),
        path,
    )


@beartype
def _write_accumulated_sequences(
    sequences_to_write: list[pl.DataFrame],
    split_path: str,
    write_format: str,
    process_id: int,
    file_index_str: str,
    target_dir: str,
    layout: StoredWindowLayout,
    col_types: dict[str, str],
):
    """Write one accumulated sequence shard."""

    if not sequences_to_write:
        return

    combined_df = pl.concat(sequences_to_write)
    curriculum_value_columns = curriculum_storage_columns(combined_df.columns)
    if curriculum_value_columns:
        combined_df = combined_df.sort(
            [
                *curriculum_value_columns,
                "sequenceId",
                "subsequenceId",
                "inputCol",
            ]
        )
    split_path_batch_seq = split_path.replace(
        f".{write_format}", f"-{process_id}-{file_index_str}.{write_format}"
    )
    out_path = insert_top_folder(split_path_batch_seq, target_dir)

    if write_format == "pt":
        process_and_write_data_pt(
            combined_df, layout.window_length, out_path, col_types
        )
    elif write_format == "parquet":
        combined_df.write_parquet(out_path)


@beartype
def _write_accumulated_windows_pt(
    windows_to_write: list[SequenceWindows],
    split_path: str,
    process_id: int,
    file_index_str: str,
    target_dir: str,
    col_types: dict[str, str],
) -> None:
    """Write one accumulated PT shard without constructing long-format rows."""
    if not windows_to_write:
        return

    split_path_batch_seq = split_path.replace(
        ".pt", f"-{process_id}-{file_index_str}.pt"
    )
    out_path = insert_top_folder(split_path_batch_seq, target_dir)
    process_and_write_sequence_windows_pt(windows_to_write, out_path, col_types)
    _write_window_shard_summary(out_path, windows_to_write)


def _window_shard_summary(windows: list[SequenceWindows]) -> dict[str, Any]:
    """Return metadata that can be recorded without rereading a shard."""
    histogram: Counter[int] = Counter()
    target_valid_from_histogram: Counter[int] = Counter()
    sample_count = 0
    for sequence in windows:
        sample_count += sequence.n_samples
        histogram.update(int(value) for value in sequence.left_pad_lengths)
        target_valid_from_histogram.update(
            max(int(left_pad), int(split_start - start))
            for left_pad, start, split_start in zip(
                sequence.left_pad_lengths,
                sequence.start_item_positions,
                sequence.split_start_item_positions,
            )
        )
    return {
        "samples": sample_count,
        "left_pad_length_histogram": {
            str(value): count for value, count in histogram.items()
        },
        "target_valid_from_histogram": {
            str(value): count for value, count in target_valid_from_histogram.items()
        },
    }


def _write_window_shard_summary(
    output_path: str, windows: list[SequenceWindows]
) -> None:
    with open(f"{output_path}.metadata.json", "w") as file:
        json.dump(_window_shard_summary(windows), file)


@beartype
def _write_accumulated_windows_parquet(
    windows_to_write: list[SequenceWindows],
    split_path: str,
    process_id: int,
    file_index_str: str,
    target_dir: str,
    schema: Any,
) -> None:
    """Write dense windows through the compatibility long-format Parquet adapter."""
    if not windows_to_write:
        return

    combined_df = _sequence_windows_list_to_long_dataframe(windows_to_write, schema)
    curriculum_value_columns = curriculum_storage_columns(combined_df.columns)
    if curriculum_value_columns:
        combined_df = combined_df.sort(
            [
                *curriculum_value_columns,
                "sequenceId",
                "subsequenceId",
                "inputCol",
            ]
        )
    split_path_batch_seq = split_path.replace(
        ".parquet", f"-{process_id}-{file_index_str}.parquet"
    )
    out_path = insert_top_folder(split_path_batch_seq, target_dir)
    combined_df.write_parquet(out_path)
    _write_window_shard_summary(out_path, windows_to_write)


class _MergedLongOutputWriters:
    """Bounded append-only worker outputs for merged CSV/Parquet preprocessing."""

    def __init__(
        self,
        split_paths: list[str],
        process_id: int,
        target_dir: str,
        write_format: str,
        schema: Any,
    ) -> None:
        self.write_format = write_format
        self.schema = schema
        self.paths = [
            insert_top_folder(
                split_path.replace(f".{write_format}", f"-{process_id}.{write_format}"),
                target_dir,
            )
            for split_path in split_paths
        ]
        self._writers: dict[int, Any] = {}
        arrow_schema = pl.DataFrame(schema=schema).to_arrow().schema
        for group, path in enumerate(self.paths):
            if write_format == "parquet":
                self._writers[group] = pq.ParquetWriter(
                    path, schema=arrow_schema, compression="snappy"
                )
            elif write_format == "csv":
                file = open(path, "wb")
                pl.DataFrame(schema=schema).write_csv(file, include_header=True)
                self._writers[group] = file
            else:
                raise ValueError(
                    f"Merged output does not support write format {write_format!r}"
                )

    def append(self, group: int, windows: list[SequenceWindows]) -> None:
        if not windows:
            return
        data = _sequence_windows_list_to_long_dataframe(windows, self.schema)
        if self.write_format == "parquet":
            self._writers[group].write_table(data.to_arrow())
        else:
            data.write_csv(self._writers[group], include_header=False)

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()


def _extract_sequence_windows_for_splits(
    batch_arrays: BatchArrays,
    run_start: int,
    run_stop: int,
    sequence_id: int,
    layout: StoredWindowLayout,
    window_stride: int,
    data_columns: list[str],
    split_ratios: Optional[list[float]],
    split_method: str,
    seed: int,
    sequence_split_assignments: Optional[dict[int, int]] = None,
    cumulative_split_ratios: Optional[np.ndarray] = None,
    alignment: Optional[PredictionAlignment] = None,
    split_values: Optional[list[Any]] = None,
) -> dict[int, Optional[SequenceWindows]]:
    """Return dense windows for one sequence across configured splits."""
    if split_method == "within_sequence":
        assert split_ratios is not None
        sequence_length = run_stop - run_start
        cumulative_ratios = (
            cumulative_split_ratios
            if cumulative_split_ratios is not None
            else np.cumsum(split_ratios)
        )
        upper_bounds = [
            int(bound) for bound in (cumulative_ratios * sequence_length).astype(int)
        ]
        lower_bounds = [0] + upper_bounds[:-1]
        sequences: dict[int, Optional[SequenceWindows]] = {}
        for i, (lb, ub) in enumerate(zip(lower_bounds, upper_bounds)):
            split_start = run_start + lb
            split_stop = run_start + ub
            if split_stop <= split_start:
                sequences[i] = None
                continue
            aligned = alignment is not None and i in alignment.splits
            context_start = run_start if aligned else split_start
            sequences[i] = _extract_sequence_windows_from_arrays(
                batch_arrays,
                context_start,
                split_stop,
                layout,
                window_stride,
                data_columns,
                split_start=split_start,
                split_stop=split_stop,
                aligned_starts=(
                    alignment.starts(
                        split_start - run_start,
                        split_stop - run_start,
                        layout.window_length,
                    )
                    if aligned and alignment is not None
                    else None
                ),
            )
        return sequences

    if split_method == "between_sequence":
        assert split_ratios is not None
        assigned_group = (
            sequence_split_assignments.get(
                sequence_id,
                assign_sequence_to_split(sequence_id, split_ratios, seed),
            )
            if sequence_split_assignments is not None
            else assign_sequence_to_split(sequence_id, split_ratios, seed)
        )
        sequences: dict[int, Optional[SequenceWindows]] = {
            i: None for i in range(len(split_ratios))
        }
        sequences[assigned_group] = _extract_sequence_windows_from_arrays(
            batch_arrays,
            run_start,
            run_stop,
            layout,
            window_stride,
            data_columns,
            aligned_starts=(
                alignment.starts(0, run_stop - run_start, layout.window_length)
                if alignment is not None and assigned_group in alignment.splits
                else None
            ),
        )
        return sequences

    if split_method == "value_cutoff":
        cutoffs = _normalized_split_cutoffs(split_values)
        if cutoffs is None or batch_arrays.split_values is None:
            raise ValueError("value_cutoff splitting requires split_values")
        values = batch_arrays.split_values[run_start:run_stop]
        if len(values) > 1 and np.any(values[1:] < values[:-1]):
            raise ValueError(
                "split_column values must be non-decreasing within each sequence"
            )
        bounds = [
            0,
            *np.searchsorted(values, cutoffs, side="left").tolist(),
            len(values),
        ]
        sequences = {}
        for i, (lower, upper) in enumerate(zip(bounds[:-1], bounds[1:])):
            split_start = run_start + int(lower)
            split_stop = run_start + int(upper)
            if split_stop <= split_start:
                sequences[i] = None
                continue
            aligned = alignment is not None and i in alignment.splits
            context_start = run_start if aligned else split_start
            sequences[i] = _extract_sequence_windows_from_arrays(
                batch_arrays,
                context_start,
                split_stop,
                layout,
                window_stride,
                data_columns,
                split_start=split_start,
                split_stop=split_stop,
                aligned_starts=(
                    alignment.starts(
                        split_start - run_start,
                        split_stop - run_start,
                        layout.window_length,
                    )
                    if aligned and alignment is not None
                    else None
                ),
            )
        return sequences

    raise ValueError(
        "split_method must be one of 'within_sequence', 'between_sequence', "
        "'value_cutoff'"
    )


@beartype
def _extract_sequences_for_splits(
    data_subset: pl.DataFrame,
    sequence_id: int,
    schema: Any,
    layout: StoredWindowLayout,
    window_stride: int,
    data_columns: list[str],
    split_ratios: Optional[list[float]],
    split_method: str,
    seed: int,
    sequence_split_assignments: Optional[dict[int, int]] = None,
    split_values: Optional[list[Any]] = None,
    alignment: Optional[PredictionAlignment] = None,
) -> dict[int, pl.DataFrame]:
    """Return long-format windows for one sequence across configured splits."""
    batch_arrays = _batch_to_arrays(data_subset, data_columns)
    windows = _extract_sequence_windows_for_splits(
        batch_arrays,
        0,
        data_subset.height,
        sequence_id,
        layout,
        window_stride,
        data_columns,
        split_ratios,
        split_method,
        seed,
        sequence_split_assignments,
        np.cumsum(split_ratios) if split_ratios is not None else None,
        alignment,
        split_values=split_values,
    )
    return {
        group: cast_columns_to_string(
            _sequence_windows_to_long_dataframe(split_windows, schema)
        )
        for group, split_windows in windows.items()
    }


@beartype
def preprocess_batch(
    project_root: str,
    data_name_root: str,
    process_id: int,
    batch: pl.DataFrame,
    schema: Any,
    split_paths: list[str],
    layout: StoredWindowLayout,
    window_stride: int,
    data_columns: list[str],
    col_types: dict[str, str],
    split_ratios: Optional[list[float]],
    target_dir: str,
    write_format: str,
    batches_per_file: int,
    merge_output: bool,
    split_method: str = "within_sequence",
    seed: int = 1010,
    sequence_split_assignments: Optional[dict[int, int]] = None,
    alignment: Optional[PredictionAlignment] = None,
    split_values: Optional[list[Any]] = None,
) -> None:
    """Extract and write all split windows for one batch."""
    batch_arrays = _batch_to_arrays(batch, data_columns)
    sequence_count = len(batch_arrays.run_starts)
    cumulative_split_ratios = (
        np.cumsum(split_ratios) if split_ratios is not None else None
    )

    def extract_for_run(
        run_start: int, run_stop: int
    ) -> dict[int, Optional[SequenceWindows]]:
        sequence_id = int(batch_arrays.sequence_ids[run_start])
        return _extract_sequence_windows_for_splits(
            batch_arrays,
            run_start,
            run_stop,
            sequence_id,
            layout,
            window_stride,
            data_columns,
            split_ratios,
            split_method,
            seed,
            sequence_split_assignments,
            cumulative_split_ratios,
            alignment,
            split_values,
        )

    if not merge_output:
        file_indices = {i: 0 for i in range(len(split_paths))}
        pad_width = len(str(max(1, sequence_count)))
        windows_by_split: dict[int, list[SequenceWindows]] = {
            i: [] for i in range(len(split_paths))
        }
        buffered_bytes = {i: 0 for i in range(len(split_paths))}

        def flush(group: int) -> None:
            if write_format == "pt":
                _write_accumulated_windows_pt(
                    windows_by_split[group],
                    split_paths[group],
                    process_id,
                    str(file_indices[group]).zfill(pad_width),
                    target_dir,
                    col_types,
                )
            elif write_format == "parquet":
                _write_accumulated_windows_parquet(
                    windows_by_split[group],
                    split_paths[group],
                    process_id,
                    str(file_indices[group]).zfill(pad_width),
                    target_dir,
                    schema,
                )

        for run_start, run_stop in zip(batch_arrays.run_starts, batch_arrays.run_stops):
            windows = extract_for_run(int(run_start), int(run_stop))
            for group, split_windows in windows.items():
                if split_windows is not None and split_windows.n_samples > 0:
                    windows_by_split[group].append(split_windows)
                    buffered_bytes[group] += _sequence_windows_nbytes(split_windows)

                if (
                    len(windows_by_split[group]) >= batches_per_file
                    or buffered_bytes[group] >= MAX_WINDOW_BUFFER_BYTES
                ):
                    flush(group)
                    windows_by_split[group] = []
                    buffered_bytes[group] = 0
                    file_indices[group] += 1

        for group in range(len(split_paths)):
            flush(group)
        return

    writers = _MergedLongOutputWriters(
        split_paths,
        process_id,
        target_dir,
        write_format,
        schema,
    )
    windows_by_split = {i: [] for i in range(len(split_paths))}
    buffered_bytes = {i: 0 for i in range(len(split_paths))}
    try:
        for run_start, run_stop in zip(batch_arrays.run_starts, batch_arrays.run_stops):
            windows = extract_for_run(int(run_start), int(run_stop))
            for group, split_windows in windows.items():
                if split_windows is not None and split_windows.n_samples > 0:
                    windows_by_split[group].append(split_windows)
                    buffered_bytes[group] += _sequence_windows_nbytes(split_windows)
                if (
                    len(windows_by_split[group]) >= batches_per_file
                    or buffered_bytes[group] >= MAX_WINDOW_BUFFER_BYTES
                ):
                    writers.append(group, windows_by_split[group])
                    windows_by_split[group] = []
                    buffered_bytes[group] = 0

        for group in range(len(split_paths)):
            writers.append(group, windows_by_split[group])
    finally:
        writers.close()


def _extract_sequence_windows_from_arrays(
    batch: BatchArrays,
    start: int,
    stop: int,
    layout: StoredWindowLayout,
    stride_for_split: int,
    columns: list[str],
    *,
    split_start: Optional[int] = None,
    split_stop: Optional[int] = None,
    aligned_starts: Optional[np.ndarray] = None,
) -> Optional[SequenceWindows]:
    """Extract dense feature windows from one known sequence."""
    if stop <= start:
        return None

    split_start = start if split_start is None else split_start
    split_stop = stop if split_stop is None else split_stop
    if not start <= split_start < split_stop <= stop:
        raise ValueError(
            "Split target bounds must be non-empty and contained in the extracted "
            "context slice."
        )

    sequence_id = int(batch.sequence_ids[start])
    curriculum_value_columns = tuple(batch.curriculum_values)
    sequence_length = stop - start
    pad_length = (
        0
        if aligned_starts is not None
        else max(0, layout.window_length - sequence_length)
    )
    subsequence_starts = (
        aligned_starts
        if aligned_starts is not None
        else get_subsequence_starts(
            sequence_length + pad_length,
            layout.window_length,
            stride_for_split,
        )
    )

    start_differences = subsequence_starts[1:] - subsequence_starts[:-1]
    if aligned_starts is None and not np.all(start_differences <= stride_for_split):
        raise ValueError(
            f"Diff of {subsequence_starts = }, {start_differences = } larger "
            f"than {stride_for_split = }"
        )

    unpadded_starts = subsequence_starts.astype(np.int64, copy=False) - pad_length
    left_pad_lengths = np.maximum(0, -unpadded_starts).astype(np.int64)
    values: dict[str, np.ndarray] = {}
    for column in columns:
        feature_values = batch.values[column][start:stop]
        if aligned_starts is not None:
            values[column] = np.stack(
                [
                    np.pad(
                        feature_values[
                            max(0, int(raw_start)) : int(raw_start)
                            + layout.window_length
                        ],
                        (int(max(0, -raw_start)), 0),
                        mode="constant",
                        constant_values=0,
                    )
                    for raw_start in unpadded_starts
                ]
            )
            continue
        if pad_length:
            feature_values = np.pad(
                feature_values,
                (pad_length, 0),
                mode="constant",
                constant_values=0,
            )
        all_windows = np.lib.stride_tricks.sliding_window_view(
            feature_values, layout.window_length
        )
        values[column] = np.ascontiguousarray(all_windows[subsequence_starts])

    item_positions = batch.item_positions[start:stop].astype(np.int64, copy=False)
    first_item_position = int(item_positions[0])
    minimum_unpadded_start = int(unpadded_starts.min())
    if minimum_unpadded_start < 0 and first_item_position < (
        INT64_INFO.min - minimum_unpadded_start
    ):
        raise ValueError(
            "startItemPosition falls outside signed Int64 after applying left padding."
        )
    position_indices = np.maximum(unpadded_starts, 0)
    absolute_starts = item_positions[position_indices].copy()
    padded_mask = unpadded_starts < 0
    absolute_starts[padded_mask] = first_item_position + unpadded_starts[padded_mask]

    split_start_item_position = int(batch.item_positions[split_start])
    split_last_item_position = int(batch.item_positions[split_stop - 1])
    if split_last_item_position == INT64_INFO.max:
        raise ValueError("splitEndItemPosition falls outside signed Int64.")
    split_end_item_position = split_last_item_position + 1
    split_start_item_positions = np.full(
        len(subsequence_starts), split_start_item_position, dtype=np.int64
    )
    split_end_item_positions = np.full(
        len(subsequence_starts), split_end_item_position, dtype=np.int64
    )

    sample_positions = None
    if curriculum_value_columns:
        raw_starts = np.maximum(unpadded_starts, 0)
        raw_stops = np.minimum(sequence_length, unpadded_starts + layout.window_length)
        sample_position_columns = []
        curriculum_values = {
            column: batch.curriculum_values[column][start:stop]
            for column in curriculum_value_columns
        }
        for column in curriculum_value_columns:
            positions = curriculum_values[column]
            change_prefix = np.concatenate(
                (
                    np.array([0], dtype=np.int64),
                    np.cumsum(positions[1:] != positions[:-1], dtype=np.int64),
                )
            )
            transition_counts = change_prefix[raw_stops - 1] - change_prefix[raw_starts]
            invalid = np.flatnonzero(
                (raw_stops <= raw_starts) | (transition_counts > 0)
            )
            if len(invalid) > 0:
                raise ValueError(
                    "Curriculum columns must be identical within each generated "
                    f"subsequence; sequenceId={sequence_id}, "
                    f"subsequenceId={int(invalid[0])}"
                )
            sample_position_columns.append(positions[raw_starts].astype(np.int64))
        sample_positions = np.column_stack(sample_position_columns)

    return SequenceWindows(
        sequence_id=sequence_id,
        subsequence_ids=np.arange(len(subsequence_starts), dtype=np.int64),
        start_item_positions=absolute_starts,
        left_pad_lengths=left_pad_lengths,
        split_start_item_positions=split_start_item_positions,
        split_end_item_positions=split_end_item_positions,
        values=values,
        curriculum_columns=tuple(curriculum_value_columns),
        sample_positions=sample_positions,
    )


@beartype
def extract_sequence_windows(
    data: pl.DataFrame,
    layout: StoredWindowLayout,
    stride_for_split: int,
    columns: list[str],
) -> Optional[SequenceWindows]:
    """Compatibility wrapper for extracting one sequence DataFrame."""
    batch = _batch_to_arrays(data, columns)
    return _extract_sequence_windows_from_arrays(
        batch,
        0,
        data.height,
        layout,
        stride_for_split,
        columns,
    )


def _sequence_windows_to_long_dataframe(
    windows: Optional[SequenceWindows], schema: Any
) -> pl.DataFrame:
    """Convert dense sequence windows to the stored long-format schema."""
    if windows is None or windows.n_samples == 0:
        return pl.DataFrame(schema=schema)

    feature_names = list(windows.values)
    feature_count = len(feature_names)
    window_length = next(iter(windows.values.values())).shape[1]
    required_schema_columns = {
        "startItemPosition",
        "splitStartItemPosition",
        "splitEndItemPosition",
    }
    schema_columns = set(schema) if isinstance(schema, dict) else set(schema.names)
    missing_schema_columns = required_schema_columns - schema_columns
    if missing_schema_columns:
        raise ValueError(
            "Preprocessed window schema is missing required position columns: "
            f"{sorted(missing_schema_columns)}"
        )
    flattened_values = np.stack(
        [windows.values[column] for column in feature_names], axis=1
    ).reshape(windows.n_samples * feature_count, window_length)

    output: dict[str, Any] = {
        "sequenceId": np.full(
            windows.n_samples * feature_count, windows.sequence_id, dtype=np.int64
        ),
        "subsequenceId": np.repeat(windows.subsequence_ids, feature_count),
        "startItemPosition": np.repeat(windows.start_item_positions, feature_count),
        "leftPadLength": np.repeat(windows.left_pad_lengths, feature_count),
        "splitStartItemPosition": np.repeat(
            windows.split_start_item_positions, feature_count
        ),
        "splitEndItemPosition": np.repeat(
            windows.split_end_item_positions, feature_count
        ),
    }
    if windows.sample_positions is not None:
        for column_index, column in enumerate(windows.curriculum_columns):
            output[column] = np.repeat(
                windows.sample_positions[:, column_index], feature_count
            )
    output["inputCol"] = np.tile(np.asarray(feature_names), windows.n_samples)
    for offset, column in enumerate([str(i) for i in range(window_length - 1, -1, -1)]):
        output[column] = flattened_values[:, offset]

    output = {
        column: values for column, values in output.items() if column in schema_columns
    }
    return pl.DataFrame(output, schema=schema)


def _sequence_windows_list_to_long_dataframe(
    windows: list[SequenceWindows], schema: Any
) -> pl.DataFrame:
    """Convert a bounded collection of dense windows in one Polars construction."""
    if not windows:
        return pl.DataFrame(schema=schema)

    feature_names = list(windows[0].values)
    feature_count = len(feature_names)
    window_length = next(iter(windows[0].values.values())).shape[1]
    required_schema_columns = {
        "startItemPosition",
        "splitStartItemPosition",
        "splitEndItemPosition",
    }
    schema_columns = set(schema) if isinstance(schema, dict) else set(schema.names)
    missing_schema_columns = required_schema_columns - schema_columns
    if missing_schema_columns:
        raise ValueError(
            "Preprocessed window schema is missing required position columns: "
            f"{sorted(missing_schema_columns)}"
        )
    output_parts: dict[str, list[np.ndarray]] = {
        "sequenceId": [],
        "subsequenceId": [],
        "startItemPosition": [],
        "leftPadLength": [],
        "splitStartItemPosition": [],
        "splitEndItemPosition": [],
        "inputCol": [],
    }
    for column in windows[0].curriculum_columns:
        output_parts[column] = []
    value_parts = []

    for sequence in windows:
        row_count = sequence.n_samples * feature_count
        output_parts["sequenceId"].append(
            np.full(row_count, sequence.sequence_id, dtype=np.int64)
        )
        output_parts["subsequenceId"].append(
            np.repeat(sequence.subsequence_ids, feature_count)
        )
        output_parts["startItemPosition"].append(
            np.repeat(sequence.start_item_positions, feature_count)
        )
        output_parts["leftPadLength"].append(
            np.repeat(sequence.left_pad_lengths, feature_count)
        )
        output_parts["splitStartItemPosition"].append(
            np.repeat(sequence.split_start_item_positions, feature_count)
        )
        output_parts["splitEndItemPosition"].append(
            np.repeat(sequence.split_end_item_positions, feature_count)
        )
        output_parts["inputCol"].append(
            np.tile(np.asarray(feature_names), sequence.n_samples)
        )
        if sequence.sample_positions is not None:
            for column_index, column in enumerate(sequence.curriculum_columns):
                output_parts[column].append(
                    np.repeat(sequence.sample_positions[:, column_index], feature_count)
                )
        value_parts.append(
            np.stack(
                [sequence.values[column] for column in feature_names], axis=1
            ).reshape(row_count, window_length)
        )

    output: dict[str, Any] = {
        column: np.concatenate(parts) for column, parts in output_parts.items()
    }
    flattened_values = np.concatenate(value_parts)
    for offset, column in enumerate([str(i) for i in range(window_length - 1, -1, -1)]):
        output[column] = flattened_values[:, offset]
    output = {
        column: values for column, values in output.items() if column in schema_columns
    }
    return pl.DataFrame(output, schema=schema)


@beartype
def extract_sequences(
    data: pl.DataFrame,
    schema: Any,
    layout: StoredWindowLayout,
    stride_for_split: int,
    columns: list[str],
) -> pl.DataFrame:
    """Extract long-format windows from one known sequence."""
    return _sequence_windows_to_long_dataframe(
        extract_sequence_windows(
            data,
            layout,
            stride_for_split,
            columns,
        ),
        schema,
    )


@beartype
def get_subsequence_starts(
    in_context_length: int,
    window_length: int,
    stride_for_split: int,
) -> np.ndarray:
    """Return distributed window starts, including the final available start."""
    last_available_start = in_context_length - window_length
    num_subsequences = math.ceil(last_available_start / stride_for_split) + 1
    starts = np.linspace(0, last_available_start, num_subsequences, dtype=int)
    return np.unique(starts)


@beartype
def extract_subsequences(
    in_seq: dict[str, list],
    window_length: int,
    stride_for_split: int,
    columns: list[str],
) -> tuple[dict[str, list[list[Union[float, int]]]], list[int], np.ndarray]:
    """Extract padded windows plus left-pad lengths from one sequence."""
    in_seq_len = len(in_seq[columns[0]])
    pad_len = 0
    if in_seq_len < window_length:
        pad_len = window_length - in_seq_len
        in_seq = {col: ([0] * pad_len) + in_seq[col] for col in columns}
    in_context_length = len(in_seq[columns[0]])

    subsequence_starts = get_subsequence_starts(
        in_context_length,
        window_length,
        stride_for_split,
    )
    subsequence_starts_diff = subsequence_starts[1:] - subsequence_starts[:-1]
    if not np.all(subsequence_starts_diff <= stride_for_split):
        raise ValueError(
            f"Diff of {subsequence_starts = }, {subsequence_starts_diff = } larger than {stride_for_split = }"
        )

    result = {
        col: [list(in_seq[col][i : i + window_length]) for i in subsequence_starts]
        for col in columns
    }
    left_pad_lengths = [pad_len] * len(subsequence_starts)

    return result, left_pad_lengths, subsequence_starts


@beartype
def insert_top_folder(path: str, folder_name: str) -> str:
    """Insert folder_name before the basename."""
    components = os.path.split(path)
    new_components = list(components[:-1]) + [folder_name] + [components[-1]]
    return os.path.join(*new_components)


@beartype
def cast_columns_to_string(data: pl.DataFrame) -> pl.DataFrame:
    """Cast Polars column names to strings."""
    data.columns = [str(col) for col in data.columns]
    return data


@beartype
def delete_files(files: Union[list[str], dict[int, list[str]]]) -> None:
    """Delete paths from a list or split-indexed dict."""
    if isinstance(files, dict):
        files = [x for y in list(files.values()) for x in y]
    for file in files:
        os.remove(file)


@beartype
def create_file_paths_for_multiple_files1(
    project_root: str,
    target_dir: str,
    n_splits: int,
    n_batches: int,
    process_id: int,
    file_index_str: str,
    dataset_name: str,
    write_format: str,
) -> dict[int, list[str]]:
    """Return per-split temp paths for one multi-file shard."""
    files = {}
    for split in range(n_splits):
        files_for_split = [
            os.path.join(
                project_root,
                "data",
                target_dir,
                f"{dataset_name}-{process_id}-{file_index_str}-split{split}-{batch_id}.{write_format}",
            )
            for batch_id in range(n_batches)
        ]
        files[split] = files_for_split
    return files


@beartype
def create_file_paths_for_single_file(
    project_root: str,
    target_dir: str,
    n_splits: int,
    n_batches: int,
    dataset_name: str,
    write_format: str,
) -> dict[int, list[str]]:
    """Return per-split temp paths for one single-file run."""
    files = {}
    for split in range(n_splits):
        files_for_split = [
            os.path.join(
                project_root,
                "data",
                target_dir,
                f"{dataset_name}-split{split}-{core_id}.{write_format}",
            )
            for core_id in range(n_batches)
        ]
        files[split] = files_for_split
    return files


@beartype
def create_file_paths_for_multiple_files2(
    project_root: str,
    target_dir: str,
    n_splits: int,
    n_processes: int,
    n_files: dict[int, int],
    dataset_name: str,
    write_format: str,
) -> dict[int, list[str]]:
    """Return per-split intermediate paths for multi-file merge."""
    files = {}
    for split in range(n_splits):
        files_for_split = []
        for process_id in range(n_processes):
            # Match the padding used by this worker when writing its file shard.
            pad_width = len(str(n_files[process_id] - 1))
            files_for_split.extend(
                os.path.join(
                    project_root,
                    "data",
                    target_dir,
                    f"{dataset_name}-{process_id}-{str(file_index).zfill(pad_width)}-split{split}.{write_format}",
                )
                for file_index in range(n_files[process_id])
            )
        files[split] = files_for_split

    return files


@beartype
def combine_multiprocessing_outputs(
    project_root: str,
    target_dir: str,
    n_splits: int,
    input_files: dict[int, list[str]],
    dataset_name: str,
    write_format: str,
    in_target_dir: bool = False,
    pre_split_str: Optional[str] = None,
    post_split_str: Optional[str] = None,
) -> None:
    """Combine per-split intermediate files."""
    for split in range(n_splits):
        split_file_path = create_split_file_path(
            project_root,
            dataset_name,
            split,
            write_format,
            in_target_dir,
            target_dir,
            pre_split_str,
            post_split_str,
        )

        logger.info(f"writing to: {split_file_path}")
        if write_format == "csv":
            command = " ".join(
                ["csvstack"] + input_files[split] + [f"> {split_file_path}"]
            )
            result = os.system(command)
            if result != 0:
                raise RuntimeError(
                    f"Command '{command}' failed with exit code {result}"
                )
        elif write_format == "parquet":
            combine_parquet_files(input_files[split], split_file_path)


@beartype
def create_split_file_path(
    project_root: str,
    dataset_name: str,
    split: int,
    write_format: str,
    in_target_dir: bool,
    target_dir: str,
    pre_split_str: Optional[str],
    post_split_str: Optional[str],
) -> str:
    if pre_split_str is None and post_split_str is None:
        file_name = f"{dataset_name}-split{split}.{write_format}"
    elif pre_split_str is not None and post_split_str is None:
        file_name = f"{dataset_name}-{pre_split_str}-split{split}.{write_format}"
    elif post_split_str is not None and pre_split_str is None:
        file_name = f"{dataset_name}-split{split}-{post_split_str}.{write_format}"
    else:
        file_name = f"{dataset_name}-{pre_split_str}-split{split}-{post_split_str}.{write_format}"

    out_path = os.path.join(project_root, "data", file_name)
    if in_target_dir:
        out_path = insert_top_folder(out_path, target_dir)

    return out_path


@beartype
def combine_parquet_files(files: list[str], out_path: str) -> None:
    """Stream-concatenate Parquet files with the first file schema."""
    schema = pq.ParquetFile(files[0]).schema_arrow
    with pq.ParquetWriter(out_path, schema=schema, compression="snappy") as writer:
        for file in files:
            parquet_file = pq.ParquetFile(file)
            for batch in parquet_file.iter_batches(batch_size=262_144):
                writer.write_batch(batch)
