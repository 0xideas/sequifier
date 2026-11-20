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
from typing import Any, Optional, Union, cast

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from loguru import logger

from sequifier.config.depth_layout import DepthLayoutRegistryModel
from sequifier.config.image_config import ImageSourceConfig, PreconditioningImageConfig
from sequifier.config.metadata import (
    dataset_tensor_kind,
    dataset_tensor_version,
    validate_persisted_metadata_versions,
)
from sequifier.config.preprocess_config import (
    CardinalityLimitModel,
    load_preprocessor_config,
)
from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.helpers import (
    FLOAT_TYPE_ORDER,
    PANDAS_TO_TORCH_TYPES,
    StoredWindowLayout,
)
from sequifier.helpers import _highest_ranked_type as _highest_ranked_type
from sequifier.helpers import _resolve_integer_sequence_type
from sequifier.helpers import (
    _smallest_float_covering_integer_range as _smallest_float_covering_integer_range,
)
from sequifier.helpers import (
    assign_sequence_to_split,
    canonicalize_polars_dtype_name,
    is_float_dtype_name,
    is_integer_dtype_name,
    polars_dtype_from_name,
    read_data,
    resolve_unified_polars_numeric_dtype,
)
from sequifier.io.categorical import (
    HASH_BUCKET_LABEL_PREFIX,
    CategoricalCodec,
    is_hash_bucket_label,
)
from sequifier.io.images import _time_ms, validate_sources
from sequifier.io.preprocessing import (
    _finite_mean_and_std,
    create_id_map,
    get_combined_statistics,
    normalize_input_values,
)
from sequifier.io.pt_payload import StoredTensorBatch, load_pt_payload, save_pt_payload
from sequifier.io.sample_order import (
    CURRICULUM_COLUMN_PREFIX,
    curriculum_columns,
    curriculum_storage_column,
    curriculum_storage_columns,
    source_curriculum_column,
)
from sequifier.io.window_placement import (
    PredictionAlignment,
    WindowPlacement,
    get_subsequence_starts,
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
MAX_WINDOW_BUFFER_BYTES = 256 * 1024 * 1024

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
        if "min_count" in config or "max_categories" in config:
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
            if "min_count" in config:
                retained = [
                    value
                    for value, count in user_counts.items()
                    if count >= int(config["min_count"])
                ]
            else:
                retained = [
                    value
                    for value, _ in sorted(
                        user_counts.items(),
                        key=lambda item: (-item[1], _category_sort_key(item[0])),
                    )[: int(config["max_categories"])]
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
            is_hash_bucket_label(value) for value in values.to_list()
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
            alignment.prediction_length * alignment.steps
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
    """Read the canonical storage dtype mapping."""
    return _normalize_column_types(metadata.get("column_data_types"))


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
    try:
        is_numeric = dtype.is_numeric
    except AttributeError:
        return False
    return is_numeric()


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
def _resolve_unified_parquet_type(column_data_types: dict[str, str]) -> Any:
    normalized = {
        column: "Int64"
        if _is_temporal_dtype_name(dtype)
        else canonicalize_polars_dtype_name(dtype)
        for column, dtype in column_data_types.items()
    }
    resolved = resolve_unified_polars_numeric_dtype(normalized)
    types = set(normalized.values())
    floats = [dtype for dtype in types if is_float_dtype_name(dtype)]
    if not floats:
        if len(types) > 1:
            logger.warning(
                "Multiple integer column_data_types were specified for Parquet output; "
                "using a unified integer schema."
            )
    else:
        requested = max(floats, key=FLOAT_TYPE_ORDER.index)
        if len(types) > 1:
            logger.warning(
                "Multiple column_data_types were specified for Parquet output; "
                f"using unified sequence dtype {requested}."
            )
        if str(resolved) != requested:
            logger.warning(
                "An integer column_type has a range exceeding "
                f"{requested}; upgrading unified Parquet sequence dtype to {resolved}."
            )
    return resolved


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
    from sequifier.config.preprocess_config import (
        ExportMetadataConfig,
        PrepareInferenceConfig,
    )
    from sequifier.config.sequence_import_config import (
        PreparePolicyConfig,
        SequenceImportConfig,
    )

    if isinstance(config, ExportMetadataConfig):
        from sequifier.io.prepare_inference import export_policy_metadata

        export_policy_metadata(config)
    elif isinstance(config, PrepareInferenceConfig):
        from sequifier.io.prepare_inference import prepare_inference

        prepare_inference(config)
    elif isinstance(config, PreparePolicyConfig):
        from sequifier.io.prepare_policy import prepare_policy

        prepare_policy(config)
    elif isinstance(config, SequenceImportConfig):
        from sequifier.trajectory.export import import_sequences

        import_sequences(config)
    else:
        from contextlib import nullcontext

        from sequifier.io.indexed_sequences import IndexedSequenceWriter

        output = config.indexed_output_path
        writer = (
            IndexedSequenceWriter(Path(config.project_root) / output)
            if output
            else nullcontext()
        )
        with writer as indexed:
            Preprocessor(
                **config.model_dump(exclude={"run"}),
                project_root=config.project_root,
                seed=config.seed,
                num_workers=config.num_workers,
                indexed_writer=indexed,
            )
    logger.info("--- Preprocessing Complete ---")


@beartype
def _folder_input_files(
    data_path: str, read_format: str, exclude_path: str | None = None
) -> list[str]:
    """Return folder input files in a stable traversal order."""
    file_paths = []
    for root, directories, files in os.walk(data_path):
        directories.sort()
        file_paths.extend(
            os.path.join(root, file)
            for file in sorted(files)
            if file.endswith(read_format)
            and (
                exclude_path is None
                or os.path.realpath(os.path.join(root, file))
                != os.path.realpath(exclude_path)
            )
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
def _audit_folder_file(
    job: tuple[str, str, Optional[int], Optional[str], Optional[list[Any]]],
) -> tuple[int, dict[int, list[int]], Optional[list[int]]]:
    path, read_format, remaining_rows, split_column, split_values = job
    cutoffs = _normalized_split_cutoffs(split_values)
    cutoff_counts = [0] * (len(cutoffs) + 1) if cutoffs is not None else None
    columns: list[str] = list(INPUT_METADATA_COLUMNS)
    if split_column is not None:
        columns.append(split_column)
    coordinates = read_data(path, read_format, columns=_deduplicate_columns(columns))
    if remaining_rows is not None:
        coordinates = coordinates.slice(0, remaining_rows)
    if cutoff_counts is not None and cutoffs is not None:
        coordinates = _add_normalized_split_column(
            coordinates, split_column, split_values, path
        )
        cutoff_counts = _cutoff_row_counts(
            coordinates.get_column(SPLIT_VALUE_COLUMN), cutoffs
        )
    coordinates = _validate_sequence_coordinates(
        coordinates.select(list(INPUT_METADATA_COLUMNS)), path
    )
    summaries: dict[int, list[int]] = {}
    starts, stops = _sequence_run_bounds(coordinates)
    sequence_ids = coordinates.get_column("sequenceId").to_numpy()
    item_positions = coordinates.get_column("itemPosition").to_numpy()
    for run_start, run_stop in zip(starts, stops):
        start, stop = int(run_start), int(run_stop)
        summaries[int(sequence_ids[start])] = [
            int(item_positions[start]),
            int(item_positions[stop - 1]),
            stop - start,
            1,
        ]
    return coordinates.height, summaries, cutoff_counts


def _merge_folder_summaries(
    summaries: dict[int, list[int]], file_summaries: dict[int, list[int]]
) -> None:
    for sequence_id, (minimum, maximum, count, fragments) in file_summaries.items():
        previous = summaries.get(sequence_id)
        if previous is None:
            summaries[sequence_id] = [minimum, maximum, count, fragments]
        else:
            previous[0] = min(previous[0], minimum)
            previous[1] = max(previous[1], maximum)
            previous[2] += count
            previous[3] += fragments


def _validate_folder_summaries(summaries: dict[int, list[int]]) -> None:
    for sequence_id, (minimum, maximum, count, _) in summaries.items():
        if maximum - minimum + 1 != count:
            raise ValueError(
                "itemPosition must increase by one within each sequence across "
                f"folder files; sequenceId={sequence_id}, range=[{minimum}, "
                f"{maximum}], rows={count}."
            )


def _audit_folder_sequence_summaries(
    file_paths: list[str],
    read_format: str,
    max_rows: Optional[int],
    split_column: Optional[str] = None,
    split_values: Optional[list[Any]] = None,
    num_workers: int = 1,
) -> tuple[list[str], dict[int, list[int]], Optional[list[int]]]:
    """Return selected folder files and global sequence coordinate summaries."""
    summaries: dict[int, list[int]] = {}
    selected_file_paths = []
    rows_read = 0
    cutoffs = _normalized_split_cutoffs(split_values)
    cutoff_counts = [0] * (len(cutoffs) + 1) if cutoffs is not None else None
    if max_rows is None and num_workers > 1 and len(file_paths) > 1:
        jobs = [
            (path, read_format, None, split_column, split_values) for path in file_paths
        ]
        with _create_preprocess_pool(min(num_workers, len(jobs))) as pool:
            results = pool.map(_audit_folder_file, jobs)
        files_and_results = zip(file_paths, results)
    else:
        results = []
        for path in file_paths:
            remaining_rows = None if max_rows is None else max_rows - rows_read
            if remaining_rows is not None and remaining_rows <= 0:
                break
            result = _audit_folder_file(
                (path, read_format, remaining_rows, split_column, split_values)
            )
            results.append(result)
            rows_read += result[0]
        files_and_results = zip(file_paths, results)
    for path, (_, file_summaries, file_cutoff_counts) in files_and_results:
        selected_file_paths.append(path)
        _merge_folder_summaries(summaries, file_summaries)
        if cutoff_counts is not None and file_cutoff_counts is not None:
            cutoff_counts = [
                count + additional
                for count, additional in zip(cutoff_counts, file_cutoff_counts)
            ]
    _validate_folder_summaries(summaries)
    return selected_file_paths, summaries, cutoff_counts


@dataclass(frozen=True)
class _FolderMetadataJob:
    path: str
    read_format: str
    selected_columns: Optional[list[str]]
    row_limit: Optional[int]
    mask_column: Optional[str]
    curriculum_column: Optional[Union[str, list[str]]]
    split_column: Optional[str]
    split_values: Optional[list[Any]]
    column_data_types: Optional[dict[str, str]]
    precomputed_id_maps: dict[str, dict[Union[str, int], int]]
    cardinality_config: dict[str, dict[str, Any]]
    categorical_columns: Optional[list[str]]
    real_columns: Optional[list[str]]
    normalize_on_all_data: bool
    split_ratios: Optional[list[float]]
    split_method: str
    seed: int
    sequence_split_assignments: Optional[dict[int, int]]
    sequence_summaries: Optional[dict[int, list[int]]]


@dataclass
class _FolderMetadataResult:
    rows: int
    data_columns: list[str]
    source_col_types: dict[str, str]
    configured_col_types: Optional[dict[str, str]]
    has_sample_positions: bool
    sequence_summaries: dict[int, list[int]]
    cutoff_counts: Optional[list[int]]
    statistics: dict[str, dict[str, float]]
    categorical_values: dict[str, set[Any]]
    categorical_counts: dict[str, Counter]


def _fit_folder_metadata_file(job: _FolderMetadataJob) -> _FolderMetadataResult:
    data = _load_and_preprocess_data(
        job.path,
        job.read_format,
        job.selected_columns,
        job.row_limit,
        job.mask_column,
        job.curriculum_column,
        False,
        job.split_column,
        job.split_values,
    )
    file_summaries: dict[int, list[int]] = {}
    for sequence_id, minimum, maximum, count in (
        data.group_by("sequenceId")
        .agg(
            pl.col("itemPosition").min().alias("__minimum"),
            pl.col("itemPosition").max().alias("__maximum"),
            pl.len().alias("__count"),
        )
        .iter_rows()
    ):
        file_summaries[int(sequence_id)] = [int(minimum), int(maximum), int(count), 1]
    cutoffs = (
        _normalized_split_cutoffs(job.split_values)
        if job.split_method == "value_cutoff"
        else None
    )
    cutoff_counts = (
        _cutoff_row_counts(data.get_column(SPLIT_VALUE_COLUMN), cutoffs)
        if cutoffs is not None
        else None
    )
    data_columns = _get_data_columns(data, job.mask_column)
    _validate_declared_column_roles(
        data_columns, job.categorical_columns, job.real_columns
    )
    configured_col_types = _configured_column_types_for_data_columns(
        job.column_data_types, data_columns
    )
    data = _apply_configured_input_casting(data, data_columns, configured_col_types)
    source_col_types = {col: str(data.schema[col]) for col in data_columns}
    for col in job.precomputed_id_maps:
        if col not in data_columns:
            raise ValueError(f"Precomputed column {col} not found in {job.path}")
    _validate_cardinality_columns(
        job.cardinality_config, data_columns, job.precomputed_id_maps
    )
    _validate_cardinality_reserved_values(data, job.cardinality_config)
    fitting_data = (
        data
        if job.normalize_on_all_data
        else _fit_data_for_split_zero(
            data,
            job.split_ratios,
            job.split_method,
            job.seed,
            job.sequence_split_assignments,
            job.sequence_summaries or file_summaries,
            job.split_values,
        )
    )
    categorical_values: dict[str, set[Any]] = {}
    categorical_counts: dict[str, Counter] = {}
    _, statistics = _get_column_statistics(
        fitting_data,
        data_columns,
        {},
        {},
        0,
        job.precomputed_id_maps,
        job.mask_column,
        categorical_values,
        categorical_columns=job.categorical_columns,
        real_columns=job.real_columns,
        categorical_value_counts=categorical_counts,
        cardinality_config=job.cardinality_config,
    )
    return _FolderMetadataResult(
        data.height,
        data_columns,
        source_col_types,
        configured_col_types,
        bool(curriculum_storage_columns(data.columns)),
        file_summaries,
        cutoff_counts,
        statistics,
        categorical_values,
        categorical_counts,
    )


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

    indexed_writer: Any = None

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
        window_stride: int,
        max_rows: Optional[int],
        seed: int,
        num_workers: Optional[int],
        batches_per_file: int,
        process_by_file: bool,
        prediction_aligned_splits: list[int],
        prediction_length: Optional[int],
        target_offset: Optional[int],
        use_precomputed_maps: Optional[list[str]],
        metadata_config_path: Optional[str],
        prediction_alignment_steps: int = 1,
        window_lengths: Optional[list[int]] = None,
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
        static_data_path: Optional[str] = None,
        static_categorical_columns: Optional[list[str]] = None,
        static_real_columns: Optional[list[str]] = None,
        image_sources: Optional[dict[str, Any]] = None,
        preconditioning_image: Optional[dict[str, Any]] = None,
        time_alignment_column: Optional[str] = None,
        indexed_writer: Any = None,
        indexed_prediction: bool = False,
        indexed_passthrough: Optional[list[str]] = None,
    ):
        """Initialize and run preprocessing from validated config fields."""
        self.depth_layouts = DepthLayoutRegistryModel.model_validate(
            depth_layouts or {}
        )
        self.project_root = project_root
        self.indexed_writer = indexed_writer
        self.indexed_prediction = indexed_prediction
        self.indexed_passthrough = indexed_passthrough or []
        passthrough = set(self.indexed_passthrough)
        reserved = {
            *INPUT_METADATA_COLUMNS,
            mask_column,
            split_column,
            time_alignment_column,
            *curriculum_columns(curriculum_column),
            *(layout.position_column for _, layout in self.depth_layouts.items()),
            *self.depth_layouts.deep_columns,
        }
        if len(passthrough) != len(self.indexed_passthrough) or any(
            not column or column in reserved for column in passthrough
        ):
            raise ValueError("Indexed passthrough must name distinct auxiliary columns")
        if selected_columns and (
            conflicts := passthrough.intersection(selected_columns)
        ):
            raise ValueError(
                f"Columns cannot be selected features and passthrough: {sorted(conflicts)}"
            )
        self.indexed_metadata: dict[str, Any] = {}
        self.batches_per_file = batches_per_file
        self.preprocessing_data_path = preprocessing_data_path
        self.static_data_path = static_data_path
        self.static_categorical_columns = static_categorical_columns or []
        self.static_real_columns = static_real_columns or []
        self.image_sources = {
            name: ImageSourceConfig.model_validate(source)
            for name, source in (image_sources or {}).items()
        }
        self.time_alignment_column = time_alignment_column
        self.image_time_rows: list[tuple[int, int, int]] = []
        self.preconditioning_image = (
            PreconditioningImageConfig.model_validate(preconditioning_image)
            if preconditioning_image is not None
            else None
        )
        self._frozen_metadata = None
        if metadata_config_path is not None:
            with open(os.path.join(project_root, metadata_config_path)) as file:
                self._frozen_metadata = json.load(file)
            validate_persisted_metadata_versions(self._frozen_metadata)
            validate_special_token_ids(
                self._frozen_metadata["special_token_ids"],
                source=f"metadata config '{metadata_config_path}'",
            )
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
        if not window_lengths:
            raise ValueError("window_lengths requires one width per split")
        resolved_window_lengths = window_lengths
        self.window_lengths: list[int] = resolved_window_lengths
        self.split_layouts = [
            StoredWindowLayout(
                width, max_target_offset, ContractVersions.STORED_WINDOW_LAYOUT
            )
            for width in resolved_window_lengths
        ]
        self.prediction_alignment_steps = prediction_alignment_steps
        self.prediction_aligned_splits = prediction_aligned_splits
        self.prediction_length = prediction_length
        self.target_offset = target_offset
        self.alignment = (
            PredictionAlignment(
                tuple(prediction_aligned_splits),
                prediction_length,
                target_offset,
                prediction_alignment_steps,
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
        frozen_metadata = (
            self._frozen_metadata if self.indexed_writer is not None else None
        )
        if frozen_metadata is not None:
            frozen_types = _column_types_from_metadata(frozen_metadata) or {}
            if any(
                frozen_types.get(column) != dtype
                for column, dtype in (self.column_data_types or {}).items()
            ):
                raise ValueError(
                    "Indexed output types must match the original metadata"
                )
            normalize_real_columns = frozen_metadata.get("normalize_real_columns", True)
        self.normalize_real_columns = normalize_real_columns
        self.normalize_on_all_data = normalize_on_all_data
        self.metadata_fitted_on_all_data = normalize_on_all_data
        if self.mask_column is not None and self.metadata_config_path is None:
            raise ValueError("metadata_config_path must be set when mask_column is set")

        self.seed = seed
        np.random.seed(seed)
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.continue_preprocessing = continue_preprocessing
        self.storage_layout = StoredWindowLayout(
            window_length=resolved_window_lengths[0],
            max_target_offset=max_target_offset,
            version=ContractVersions.STORED_WINDOW_LAYOUT,
        )
        if self.indexed_writer is not None:
            if continue_preprocessing:
                raise ValueError("Indexed preparation can be restarted from raw input")
            self.target_dir = str(self.indexed_writer.root / "scratch")
        self.has_sample_positions = False
        self.output_n_classes = {}
        self.output_column_data_types = {}
        self.image_manifest = self._validate_image_sources()
        self._setup_directories()

        if self.time_alignment_column is not None or self.indexed_passthrough:
            if selected_columns is None:
                source_file = (
                    self.preprocessing_data_path
                    if os.path.isfile(self.preprocessing_data_path)
                    else _folder_input_files(
                        self.preprocessing_data_path,
                        self.read_format,
                        self.static_data_path,
                    )[0]
                )
                source_columns = (
                    pq.read_schema(source_file).names
                    if self.read_format == "parquet"
                    else pl.scan_csv(source_file).collect_schema().names()
                )
                selected_columns = [
                    column
                    for column in source_columns
                    if column
                    not in {
                        *INPUT_METADATA_COLUMNS,
                        self.time_alignment_column,
                        self.mask_column,
                        *self.indexed_passthrough,
                    }
                ]
            else:
                selected_columns = [
                    column
                    for column in selected_columns
                    if column != self.time_alignment_column
                ]

        if selected_columns is not None:
            selected_columns = ["sequenceId", "itemPosition"] + selected_columns
            if self.mask_column is not None and self.mask_column in selected_columns:
                raise ValueError(
                    f"'{self.mask_column}' is not allowed to be in 'selected_columns'"
                )

        self._setup_split_paths(write_format, self.n_splits)
        self._static_source = self._load_static_source()
        if frozen_metadata is not None and self._static_source is not None:
            supplied = set(self._static_source.columns) - {"sequenceId"}
            expected = set(frozen_metadata.get("static_columns", []))
            if not supplied <= expected:
                raise ValueError(
                    f"Unexpected static columns: {sorted(supplied - expected)}"
                )
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
            if self.indexed_writer is not None:
                self._finish_indexed()
            else:
                self._finish_static_preprocessing()
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
                self._finish_static_preprocessing()
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
            if self._frozen_metadata is not None:
                _, id_maps, selected_columns_statistics, n_classes, col_types = (
                    self._frozen_temporal_metadata(
                        self._frozen_metadata, data_columns=data_columns
                    )
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
                id_maps,
                n_classes,
                col_types,
                selected_columns_statistics,
                data_columns=data_columns,
            )

            if self.indexed_writer is not None:
                data = self._add_indexed_passthrough(data, preprocessing_data_path)
                self._write_indexed_items(data, sequence_split_assignments)
                self._finish_indexed()
                return

            schema = self._create_schema(col_types, self.storage_layout.window_length)

            n_batches = _process_batches_single_file(
                self.project_root,
                self.data_name_root,
                data,
                schema,
                self.num_workers,
                self.split_layouts,
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
            files_to_process = _folder_input_files(
                preprocessing_data_path,
                read_format,
                self.static_data_path if self.static_data_path else None,
            )
            self.has_sample_positions = _input_has_sample_positions(
                files_to_process, read_format, self.curriculum_column
            )
            folder_sequence_summaries = None
            folder_cutoff_counts = None
            fragmented_sequence_ids: set[int] = set()
            folder_sequence_ids: list[int] = []
            combine_audit_and_fit = (
                self.metadata_config_path is None
                and not self.normalize_on_all_data
                and self.split_method == "within_sequence"
            )
            if not combine_audit_and_fit and (
                self.metadata_config_path or not self.normalize_on_all_data
            ):
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
                    self.num_workers,
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

            if self._frozen_metadata is not None:
                (
                    data_columns,
                    id_maps,
                    selected_columns_statistics,
                    n_classes,
                    col_types,
                ) = self._frozen_temporal_metadata(
                    self._frozen_metadata, selected_columns=selected_columns
                )
                assert col_types is not None
                configured_col_types = _configured_column_types_for_data_columns(
                    self.column_data_types, data_columns
                )
                _validate_declared_column_roles(
                    data_columns, self.categorical_columns, self.real_columns
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
                if combine_audit_and_fit and fragmented_sequence_ids:
                    logger.info(
                        "Sequences span files; refitting metadata with global "
                        "split boundaries."
                    )
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
                        None,
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
                id_maps,
                n_classes,
                col_types,
                selected_columns_statistics,
                data_columns=data_columns,
            )
            schema = self._create_schema(col_types, self.storage_layout.window_length)

            if fragmented_sequence_ids or self.indexed_writer is not None:
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
                    self.split_layouts,
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
                    self.num_workers,
                    self.split_layouts,
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

        if self.indexed_writer is not None:
            self._finish_indexed()
            return
        self._cleanup(write_format)
        self._finish_static_preprocessing()

    def _add_indexed_passthrough(self, data, path):
        if not self.indexed_passthrough:
            return data
        coordinates = ["sequenceId", "itemPosition"]
        auxiliary = read_data(
            path, self.read_format, columns=coordinates + self.indexed_passthrough
        ).with_columns(pl.col(coordinates).cast(pl.Int64))
        return data.join(
            auxiliary, on=coordinates, how="left", validate="1:1", maintain_order="left"
        )

    def _write_indexed_items(self, data, assignments):
        arrays = _batch_to_arrays(data, [])
        self.indexed_metadata["indexed_curriculum_columns"] = list(
            arrays.curriculum_values
        )
        for start, stop in zip(arrays.run_starts.tolist(), arrays.run_stops.tolist()):
            sequence = int(arrays.sequence_ids[start])
            views = (
                {}
                if self.indexed_prediction
                else _place_sequence_windows_for_splits(
                    arrays,
                    start,
                    stop,
                    sequence,
                    self.split_layouts,
                    self.window_stride,
                    self.split_ratios,
                    self.split_method,
                    self.seed,
                    assignments,
                    alignment=self.alignment,
                    split_values=self.split_values,
                )
            )
            self.indexed_writer.add(
                data.slice(start, stop - start).drop(SPLIT_VALUE_COLUMN, strict=False),
                views,
            )

    def _finish_indexed(self):
        from sequifier.trajectory.sources import file_checksum

        if self._static_source is not None:
            self._finalize_static_source()
        shutil.rmtree(Path(self.project_root) / "data" / self.target_dir)
        original = (
            Path(self.project_root) / self.metadata_config_path
            if self.metadata_config_path
            else None
        )
        input_path = Path(self.preprocessing_data_path)
        paths = (
            [input_path]
            if input_path.is_file()
            else [
                Path(path)
                for path in _folder_input_files(
                    str(input_path), self.read_format, self.static_data_path
                )
            ]
        )
        if self.static_data_path:
            paths.append(Path(self.static_data_path))
        self.indexed_writer.publish(
            self.indexed_metadata,
            feature_columns=self.indexed_feature_columns,
            original_metadata=original,
            provenance={
                "source": {
                    "preprocessing_data_path": str(input_path.absolute()),
                    "static_data_path": str(Path(self.static_data_path).absolute())
                    if self.static_data_path
                    else None,
                    "indexed_passthrough": self.indexed_passthrough,
                },
                "files": [
                    {"path": str(path.absolute()), "sha256": file_checksum(path)}
                    for path in paths
                ],
            },
        )

    def _finish_static_preprocessing(self) -> None:
        """Publish static metadata before discarding resumable temp state."""
        if self._static_source is None:
            return
        self._finalize_static_source()
        shutil.rmtree(Path(self.project_root) / "data" / self.target_dir)

    def _validate_image_sources(self) -> dict[str, Any]:
        if not self.image_sources and self.preconditioning_image is None:
            return {}
        if self.depth_layouts:
            files = (
                [self.preprocessing_data_path]
                if os.path.isfile(self.preprocessing_data_path)
                else _folder_input_files(
                    self.preprocessing_data_path,
                    self.read_format,
                    self.static_data_path,
                )
            )
            positions_by_sequence: dict[int, set[int]] = {}
            rows_read = 0
            for file in files:
                remaining = None if self.max_rows is None else self.max_rows - rows_read
                if remaining is not None and remaining <= 0:
                    break
                coordinates = read_data(
                    file, self.read_format, columns=list(INPUT_METADATA_COLUMNS)
                )
                if remaining is not None:
                    coordinates = coordinates.slice(0, remaining)
                rows_read += coordinates.height
                for sequence_id, item in coordinates.iter_rows():
                    if (
                        isinstance(sequence_id, bool)
                        or isinstance(item, bool)
                        or not isinstance(sequence_id, int)
                        or not isinstance(item, int)
                    ):
                        raise ValueError("Image sequence coordinates must be integers")
                    positions_by_sequence.setdefault(sequence_id, set()).add(item)
            summaries = {
                sequence_id: [min(items), max(items), len(items), 1]
                for sequence_id, items in positions_by_sequence.items()
            }
            _validate_folder_summaries(summaries)
        elif os.path.isfile(self.preprocessing_data_path):
            _, summaries, _ = _audit_folder_file(
                (
                    self.preprocessing_data_path,
                    self.read_format,
                    self.max_rows,
                    None,
                    None,
                )
            )
        else:
            files = _folder_input_files(
                self.preprocessing_data_path,
                self.read_format,
                self.static_data_path,
            )
            _, summaries, _ = _audit_folder_sequence_summaries(
                files, self.read_format, self.max_rows, num_workers=1
            )
        if any(summary[0] != 0 for summary in summaries.values()):
            raise ValueError("Image sequences require itemPosition to start at zero")
        item_times = None
        time_mode = None
        time_index_path = None
        if self.time_alignment_column is not None:
            files = (
                [self.preprocessing_data_path]
                if os.path.isfile(self.preprocessing_data_path)
                else _folder_input_files(
                    self.preprocessing_data_path,
                    self.read_format,
                    self.static_data_path,
                )
            )
            rows_read = 0
            values_by_sequence: dict[int, dict[int, int]] = {}
            for file in files:
                remaining = None if self.max_rows is None else self.max_rows - rows_read
                if remaining is not None and remaining <= 0:
                    break
                data = read_data(
                    file,
                    self.read_format,
                    columns=["sequenceId", "itemPosition", self.time_alignment_column],
                )
                if remaining is not None:
                    data = data.slice(0, remaining)
                rows_read += data.height
                for sequence_id, position, value in data.iter_rows():
                    if value is None:
                        raise ValueError("time_alignment_column cannot contain null")
                    if isinstance(value, (date, datetime)):
                        representation = "absolute"
                    elif isinstance(value, int) and not isinstance(value, bool):
                        representation = (
                            "absolute" if len(str(value)) == 17 else "relative"
                        )
                    elif isinstance(value, str):
                        representation = (
                            "relative"
                            if len(value) == 9 and value.isdigit()
                            else "absolute"
                        )
                    else:
                        raise ValueError(
                            f"Invalid time_alignment_column value: {value!r}"
                        )
                    if time_mode is not None and time_mode != representation:
                        raise ValueError(
                            "time_alignment_column must use one time representation"
                        )
                    time_mode = representation
                    stamp = _time_ms(value, absolute=representation == "absolute")
                    positions = values_by_sequence.setdefault(int(sequence_id), {})
                    if int(position) in positions:
                        if self.depth_layouts and positions[int(position)] == stamp:
                            continue
                        raise ValueError("Duplicate item time coordinate")
                    positions[int(position)] = stamp
            item_times = {}
            for sequence_id, length in (
                (sid, summary[2]) for sid, summary in summaries.items()
            ):
                positions = values_by_sequence.get(sequence_id, {})
                if set(positions) != set(range(length)):
                    raise ValueError(
                        f"Incomplete time alignment for sequence {sequence_id}"
                    )
                times = [positions[item] for item in range(length)]
                if time_mode == "relative" and times[0] != 0:
                    raise ValueError("Relative item time must start at 000000000")
                if any(b <= a for a, b in zip(times, times[1:])):
                    raise ValueError(
                        f"Item times must increase for sequence {sequence_id}"
                    )
                item_times[sequence_id] = times
                self.image_time_rows.extend(
                    (sequence_id, item, stamp) for item, stamp in enumerate(times)
                )
            time_index_path = f"data/{self.data_name_root}-image-time-index.parquet"
        video_index_dir = (
            str(self.indexed_writer.root / "video-indexes")
            if self.indexed_writer is not None
            else f"data/{self.data_name_root}-video-indexes"
        )
        manifest = validate_sources(
            self.image_sources,
            self.preconditioning_image,
            {sequence_id: summary[2] for sequence_id, summary in summaries.items()},
            self.project_root,
            item_times=item_times,
            time_mode=time_mode,
            time_index_path=time_index_path,
            video_index_dir=video_index_dir,
        )
        if self.indexed_writer is not None:
            for name, source in manifest["image_sources"].items():
                source["path"] = str(
                    Path(self.project_root) / self.image_sources[name].path
                )
                paths = source.get("video_index_paths", {})
                for sequence, path in paths.items():
                    index_path = Path(path)
                    self.indexed_writer.files.append(index_path)
                    paths[sequence] = str(
                        index_path.relative_to(self.indexed_writer.root)
                    )
            if prefix := manifest.get("preconditioning_image"):
                assert self.preconditioning_image is not None
                prefix["paths"] = {
                    name: str(Path(self.project_root) / path)
                    for name, path in self.preconditioning_image.paths.items()
                }
        manifest["time_alignment_column"] = self.time_alignment_column
        return manifest

    def _load_static_source(self) -> Optional[pl.DataFrame]:
        """Validate the one-to-one source table before writing any windows."""
        if self.static_data_path is None:
            return None
        path = self.static_data_path
        static = (
            pl.read_parquet(path)
            if path.lower().endswith(".parquet")
            else pl.read_csv(path)
        )
        if "sequenceId" not in static.columns:
            raise ValueError("Static data must contain sequenceId")
        if not is_integer_dtype_name(str(static.schema["sequenceId"])):
            raise ValueError("Static sequenceId values must use an integer dtype")
        static = static.with_columns(pl.col("sequenceId").cast(pl.Int64, strict=True))
        if (
            static.get_column("sequenceId").null_count()
            or static.get_column("sequenceId").n_unique() != static.height
        ):
            raise ValueError("Static data requires one non-null row per sequenceId")
        columns = [column for column in static.columns if column != "sequenceId"]
        if not columns:
            raise ValueError("Static data must contain at least one feature column")
        declared = set(self.static_categorical_columns + self.static_real_columns)
        missing_columns = declared - set(columns)
        if missing_columns:
            raise ValueError(
                f"Static column roles reference missing columns: {sorted(missing_columns)}"
            )
        for column in columns:
            values = static.get_column(column)
            if values.null_count() or (
                isinstance(values.dtype, (pl.Float16, pl.Float32, pl.Float64))
                and not values.is_finite().all()
            ):
                raise ValueError(
                    f"Static column {column!r} must contain finite, non-null values"
                )
        source_files = (
            [self.preprocessing_data_path]
            if os.path.isfile(self.preprocessing_data_path)
            else _folder_input_files(
                self.preprocessing_data_path,
                self.read_format,
                self.static_data_path if self.static_data_path else None,
            )
        )
        sequence_ids: set[int] = set()
        for source in source_files:
            scan = (
                pl.scan_parquet(source)
                if self.read_format == "parquet"
                else pl.scan_csv(source)
            )
            source_schema = scan.collect_schema()
            if not is_integer_dtype_name(str(source_schema["sequenceId"])):
                raise ValueError(
                    f"Sequence data {source!r} must use integer sequenceId values"
                )
            overlap = set(columns) & set(source_schema.names())
            if overlap:
                raise ValueError(
                    f"Static columns also occur in sequence data: {sorted(overlap)}"
                )
            sequence_ids.update(
                scan.select(pl.col("sequenceId").cast(pl.Int64, strict=True).unique())
                .collect()
                .get_column("sequenceId")
                .to_list()
            )
        if None in sequence_ids:
            raise ValueError("Sequence data contains a null sequenceId")
        static_ids = set(static.get_column("sequenceId").to_list())
        missing = sequence_ids - static_ids
        extra = static_ids - sequence_ids
        if missing or extra:
            raise ValueError(
                "Static and sequence data must have identical sequenceId sets; "
                f"missing in static: {sorted(missing)[:10]}; "
                f"extra in static: {sorted(extra)[:10]}"
            )
        return static.sort("sequenceId")

    def _training_sequence_ids(self) -> set[int]:
        if self.indexed_writer is not None:
            return self.indexed_writer.split_sequence_ids(0)
        path = self.split_paths[0]
        if os.path.isdir(os.path.splitext(path)[0]):
            folder = os.path.splitext(path)[0]
            files = sorted(Path(folder).glob(f"*.{self.write_format}"))
        else:
            files = [Path(path)]
        ids: set[int] = set()
        for file in files:
            if self.write_format == "pt":
                payload = load_pt_payload(file, layouts=self.depth_layouts)
                ids.update(payload.sequence_ids.tolist())
            else:
                scan = (
                    pl.scan_parquet(file)
                    if self.write_format == "parquet"
                    else pl.scan_csv(file)
                )
                ids.update(
                    scan.select(pl.col("sequenceId").cast(pl.Int64).unique())
                    .collect()
                    .get_column("sequenceId")
                    .to_list()
                )
        return ids

    def _finalize_static_source(self) -> None:
        """Fit static transforms once per sequence and persist one feature table."""
        assert self._static_source is not None
        static = self._static_source
        columns = [column for column in static.columns if column != "sequenceId"]
        existing = self._frozen_metadata
        if existing is not None and set(columns) <= set(
            existing.get("static_columns", [])
        ):
            maps = {
                column: existing["id_maps"][column]
                for column in columns
                if column in existing["id_maps"]
            }
            stats = {
                column: existing["selected_columns_statistics"][column]
                for column in columns
                if column in existing["selected_columns_statistics"]
            }
            classes = {column: existing["n_classes"][column] for column in maps}
            types = {
                column: existing["column_data_types"][column] for column in columns
            }
            _validate_declared_roles_against_metadata(
                self.static_categorical_columns,
                self.static_real_columns,
                maps,
                stats,
                types,
            )
        else:
            fitting = (
                static
                if self.normalize_on_all_data
                else static.filter(
                    pl.col("sequenceId").is_in(self._training_sequence_ids())
                )
            )
            maps, stats = _get_column_statistics(
                fitting,
                columns,
                {},
                {},
                0,
                {},
                categorical_columns=self.static_categorical_columns,
                real_columns=self.static_real_columns,
            )
            classes = None
            types = None
        static, classes, types = _apply_column_statistics(
            static,
            columns,
            maps,
            stats,
            self.normalize_real_columns,
            n_classes=classes,
            col_types=types,
        )
        static = _apply_output_type_casting(static, columns, types)
        relative_path = os.path.join("data", f"{self.data_name_root}-static.parquet")
        if self.indexed_writer is None:
            static.write_parquet(os.path.join(self.project_root, relative_path))
        else:
            self.indexed_writer.write_static(static)
        self.output_n_classes.update(classes)
        self.output_column_data_types.update(types)
        self.output_id_maps.update(maps)
        self.output_selected_columns_statistics.update(
            {
                column: {"mean": value["mean"], "std": value["std"]}
                for column, value in stats.items()
            }
        )
        if self.indexed_writer is not None:
            self.indexed_metadata.update(
                static_columns=columns,
                static_features_path="static.parquet",
                n_classes=self.output_n_classes,
                column_data_types=self.output_column_data_types,
                id_maps=self.output_id_maps,
                selected_columns_statistics=self.output_selected_columns_statistics,
            )
            return
        metadata_files = [
            Path(self.project_root)
            / "configs"
            / "metadata_configs"
            / f"{self.data_name_root}.json"
        ]
        if not self.merge_output:
            metadata_files.extend(
                Path(os.path.splitext(path)[0]) / "metadata.json"
                for path in self.split_paths
            )
        for path in metadata_files:
            with open(path) as file:
                metadata = json.load(file)
            metadata["static_columns"] = columns
            metadata["static_features_path"] = relative_path
            metadata["n_classes"].update(classes)
            metadata["column_data_types"].update(types)
            metadata["id_maps"].update(maps)
            metadata["selected_columns_statistics"].update(
                self.output_selected_columns_statistics
            )
            with open(path, "w") as file:
                json.dump(metadata, file, indent=4)

    def _frozen_temporal_metadata(
        self, metadata, *, data_columns=None, selected_columns=None
    ):
        static = set(metadata.get("static_columns", ()))

        def temporal(mapping):
            return {key: value for key, value in mapping.items() if key not in static}

        id_maps = temporal(metadata["id_maps"])
        statistics = temporal(metadata["selected_columns_statistics"])
        n_classes = temporal(metadata["n_classes"])
        types = _column_types_from_metadata(metadata)
        if types is not None:
            types = temporal(types)
        if data_columns is None:
            if types is None:
                raise ValueError(
                    "Metadata used with folder preprocessing must contain 'column_data_types' field."
                )
            reserved = _reserved_input_columns(self.mask_column)
            data_columns = [
                column
                for column in types
                if column not in reserved
                and (not self.indexed_prediction or column in (selected_columns or ()))
            ]

        self.metadata_fitted_on_all_data = metadata.get("normalize_on_all_data", True)
        self._use_metadata_cardinality_config(metadata)
        if self.indexed_prediction:
            self.cardinality_config = {
                key: value
                for key, value in self.cardinality_config.items()
                if key in data_columns
            }
        _validate_cardinality_columns(self.cardinality_config, data_columns, {})
        _validate_declared_roles_against_metadata(
            self.categorical_columns, self.real_columns, id_maps, statistics, types
        )
        return data_columns, id_maps, statistics, n_classes, types

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
        """Fit folder metadata with ordered reduction of per-file summaries."""

        precomputed_id_maps = load_precomputed_id_maps(
            self.project_root, None, self.use_precomputed_maps
        )
        id_maps: dict[str, dict[Union[str, int], int]] = {}
        selected_columns_statistics: dict[str, dict[str, float]] = {}
        col_types: Optional[dict[str, str]] = None
        data_columns: Optional[list[str]] = None
        sample_position_presence: Optional[bool] = None
        selected_file_paths: list[str] = []
        coordinate_summaries: dict[int, list[int]] = {}
        cutoffs = (
            _normalized_split_cutoffs(self.split_values)
            if self.split_method == "value_cutoff"
            else None
        )
        cutoff_counts = [0] * (len(cutoffs) + 1) if cutoffs is not None else None
        categorical_value_sets: dict[str, set[Any]] = {}
        categorical_value_counts: dict[str, Counter] = {}
        rows_read = 0

        def job_for(path: str, row_limit: Optional[int]) -> _FolderMetadataJob:
            return _FolderMetadataJob(
                path=path,
                read_format=read_format,
                selected_columns=selected_columns,
                row_limit=row_limit,
                mask_column=self.mask_column,
                curriculum_column=self.curriculum_column,
                split_column=self.split_column,
                split_values=self.split_values,
                column_data_types=column_data_types,
                precomputed_id_maps=precomputed_id_maps,
                cardinality_config=self.cardinality_config,
                categorical_columns=self.categorical_columns,
                real_columns=self.real_columns,
                normalize_on_all_data=self.normalize_on_all_data,
                split_ratios=self.split_ratios,
                split_method=self.split_method,
                seed=self.seed,
                sequence_split_assignments=sequence_split_assignments,
                sequence_summaries=sequence_summaries,
            )

        def consume(path: str, result: _FolderMetadataResult) -> None:
            nonlocal col_types, data_columns, sample_position_presence
            nonlocal cutoff_counts, rows_read
            selected_file_paths.append(path)
            rows_read += result.rows
            _merge_folder_summaries(coordinate_summaries, result.sequence_summaries)
            if cutoff_counts is not None and result.cutoff_counts is not None:
                cutoff_counts = [
                    count + additional
                    for count, additional in zip(cutoff_counts, result.cutoff_counts)
                ]
            if sample_position_presence is None:
                sample_position_presence = result.has_sample_positions
            elif sample_position_presence != result.has_sample_positions:
                raise ValueError(
                    "The curriculum column must be present in every input file or none"
                )
            if col_types is None:
                data_columns = result.data_columns
                col_types = result.configured_col_types or result.source_col_types
            else:
                if set(result.data_columns) != set(col_types):
                    missing = set(col_types) - set(result.data_columns)
                    extra = set(result.data_columns) - set(col_types)
                    raise ValueError(
                        f"Schema mismatch in file '{path}'.\n"
                        f"Expected columns: {list(col_types)}\n"
                        f"Found columns: {result.data_columns}\n"
                        f"Missing: {missing}\n"
                        f"Extra: {extra}"
                    )
                if column_data_types is None:
                    for col in result.data_columns:
                        if result.source_col_types[col] != col_types[col]:
                            raise ValueError(
                                f"Type mismatch for column '{col}' in file '{path}'. "
                                f"Expected {col_types[col]}, got "
                                f"{result.source_col_types[col]}"
                            )
            for column, values in result.categorical_values.items():
                categorical_value_sets.setdefault(column, set()).update(values)
            for column, counts in result.categorical_counts.items():
                categorical_value_counts.setdefault(column, Counter()).update(counts)
            for column, chunk in result.statistics.items():
                previous = selected_columns_statistics.get(column)
                if previous is None:
                    selected_columns_statistics[column] = dict(chunk)
                else:
                    chunk_count = int(chunk["count"])
                    previous_count = int(previous["count"])
                    mean, std = get_combined_statistics(
                        chunk_count,
                        chunk["mean"],
                        chunk["std"],
                        previous_count,
                        previous["mean"],
                        previous["std"],
                    )
                    selected_columns_statistics[column] = {
                        "mean": mean,
                        "std": std,
                        "count": float(chunk_count + previous_count),
                    }

        logger.info(f"Data path: {data_path}")
        if max_rows is None and self.num_workers > 1 and len(files_to_process) > 1:
            jobs = (job_for(path, None) for path in files_to_process)
            with _create_preprocess_pool(
                min(self.num_workers, len(files_to_process))
            ) as pool:
                for path, result in zip(
                    files_to_process, pool.imap(_fit_folder_metadata_file, jobs)
                ):
                    consume(path, result)
        else:
            for path in files_to_process:
                row_limit = None if max_rows is None else max_rows - rows_read
                if row_limit is not None and row_limit <= 0:
                    break
                consume(path, _fit_folder_metadata_file(job_for(path, row_limit)))

        _validate_folder_summaries(coordinate_summaries)
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
            col: max(SPECIAL_TOKEN_IDS.user_start, max(mapping.values()) + 1)
            for col, mapping in id_maps.items()
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
            {
                sequence_id
                for sequence_id, summary in coordinate_summaries.items()
                if summary[3] > 1
            },
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
            if self.indexed_writer is None:
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
        layout: StoredWindowLayout | list[StoredWindowLayout],
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
            max(self.num_workers, math.ceil(total_input_bytes / target_bucket_bytes)),
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
            if self.indexed_writer is not None:
                data = self._add_indexed_passthrough(data, path)
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
            _create_preprocess_pool(self.num_workers)
            if self.num_workers > 1 and self.indexed_writer is None
            else None
        )
        try:
            for bucket, paths in sorted(bucket_files.items()):
                data = _validate_sequence_coordinates(
                    pl.concat([pl.read_parquet(path) for path in paths]),
                    f"fragment bucket {bucket}",
                )
                if self.indexed_writer is not None:
                    self._write_indexed_items(data, sequence_split_assignments)
                    continue
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
                    self.num_workers,
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

        if self.merge_output and self.indexed_writer is None:
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
        num_workers: int,
        layout: StoredWindowLayout | list[StoredWindowLayout],
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
                num_workers=num_workers,
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
            worker_count = min(num_workers, len(file_paths))
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
                "num_workers": 1,
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
                "curriculum_column": self.curriculum_column,
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
            logger.info(f"_process_batches_multiple_files num_workers: {num_workers}")
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
        """Move split outputs and write folder metadata."""

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

        if self._static_source is None:
            shutil.rmtree(directory)

    @beartype
    def _layout_metadata(self) -> dict[str, Any]:
        tensor_kind = dataset_tensor_kind(self.depth_layouts)
        return {
            "depth_layouts": self.depth_layouts.model_dump(mode="json"),
            "tensor_payload_kind": tensor_kind,
            "tensor_payload_version": dataset_tensor_version(tensor_kind),
            "window_length": self.storage_layout.window_length,
            "window_lengths": self.window_lengths,
            "prediction_alignment_steps": self.prediction_alignment_steps,
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
            "manifest_version": ContractVersions.PREPROCESSING_MANIFEST,
            "preprocessing_config": {
                **self._layout_metadata(),
                "read_format": self.read_format,
                "static_data_path": self.static_data_path,
                "static_categorical_columns": self.static_categorical_columns,
                "static_real_columns": self.static_real_columns,
                **self.image_manifest,
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
                    if self.curriculum_column is not None
                    else {}
                ),
                "use_precomputed_maps": self.use_precomputed_maps,
                "cardinality_config": self.cardinality_config,
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
            require_contract_version(
                "preprocessing manifest",
                previous_manifest.get("manifest_version"),
                ContractVersions.PREPROCESSING_MANIFEST,
            )
            validate_persisted_metadata_versions(
                previous_manifest["preprocessing_config"]
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
        require_contract_version(
            "preprocessing manifest",
            manifest.get("manifest_version"),
            ContractVersions.PREPROCESSING_MANIFEST,
        )
        previous = manifest["preprocessing_config"]
        validate_persisted_metadata_versions(previous)
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
        self.output_id_maps = previous["id_maps"]
        self.output_selected_columns_statistics = previous[
            "selected_columns_statistics"
        ]
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
        *,
        data_columns: list[str] | None = None,
    ) -> None:
        """Write metadata config JSON for training/inference."""
        if self.image_time_rows:
            if self.indexed_writer is not None:
                time_index_path = self.indexed_writer.root / "item-times.parquet"
                self.indexed_writer.files.append(time_index_path)
                for source in self.image_manifest.get("image_sources", {}).values():
                    if source.get("time_index_path"):
                        source["time_index_path"] = time_index_path.name
            else:
                time_index_path = (
                    Path(self.project_root)
                    / "data"
                    / f"{self.data_name_root}-image-time-index.parquet"
                )
            pl.DataFrame(
                self.image_time_rows,
                schema=["sequenceId", "itemPosition", "timeMs"],
                orient="row",
            ).write_parquet(time_index_path)
        self.output_n_classes = n_classes
        self.output_column_data_types = col_types
        self.output_id_maps = id_maps
        self.output_selected_columns_statistics = {
            col: {"mean": stats["mean"], "std": stats["std"]}
            for col, stats in selected_columns_statistics.items()
        }
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
            **self.image_manifest,
            "window_stride": self.window_stride,
            "raw_mask_column": self.mask_column,
            "raw_curriculum_column": self.curriculum_column,
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
        if self.indexed_writer is not None:
            self.indexed_feature_columns = data_columns or list(col_types)
            names = (
                ["predict"]
                if self.indexed_prediction
                else [str(index) for index in range(self.n_splits)]
            )
            data_driven_config.update(
                split_paths=[f"{name}.split.json" for name in names],
                indexed_views=names,
                indexed_passthrough=self.indexed_passthrough,
                fit_split_indices=(
                    []
                    if self.metadata_config_path
                    else (
                        list(range(self.n_splits))
                        if self.normalize_on_all_data
                        else [0]
                    )
                ),
            )
            self.indexed_metadata.update(data_driven_config)
            return
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
                        n_classes=self.output_n_classes,
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
            "n_classes": self.output_n_classes,
            "column_data_types": self.output_column_data_types,
            "id_maps": self.output_id_maps,
            "selected_columns_statistics": self.output_selected_columns_statistics,
            "normalize_real_columns": self.normalize_real_columns,
            **self.image_manifest,
            "total_samples": total_samples,
            "batch_files": batch_files_metadata,
            **self._layout_metadata(),
        }
        metadata["window_length"] = self.window_lengths[split_index]
        if self.has_sample_positions:
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
            codec = CategoricalCodec(
                id_maps[col], data.schema[col], (cardinality_config or {}).get(col)
            )
            expressions.append(codec.expression(col))
            if not col_types_was_provided:
                col_types[col] = "Int64"
        elif col in selected_columns_statistics:
            if not col_types_was_provided and not is_float_dtype_name(col_types[col]):
                col_types[col] = "Float64"
            if normalize_real_columns:
                expressions.append(
                    cast(
                        pl.Expr,
                        normalize_input_values(
                            pl.col(col),
                            selected_columns_statistics[col],
                            col_types[col],
                        ),
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
                    if "min_count" in cardinality or "max_categories" in cardinality:
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
def _validate_sequence_coordinates(
    data: pl.DataFrame, source: str, sort_output: bool = True
) -> pl.DataFrame:
    """Sort once and reject ambiguous or unrepresentable coordinates."""
    for column in INPUT_METADATA_COLUMNS:
        assert column in data.schema, f"{source} is missing {column}."
        dtype = data.schema[column]
        assert (
            dtype.is_integer()
        ), f"{column} must have an integer dtype in {source}; found {dtype}."
        values = data.get_column(column)
        assert values.null_count() == 0, f"{column} contains null values in {source}."
        if not values.is_empty():
            minimum, maximum = int(values.min()), int(values.max())
            assert INT64_INFO.min <= minimum and maximum <= INT64_INFO.max, (
                f"{column} must fit in signed Int64 in {source}; found range "
                f"[{minimum}, {maximum}]."
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
    num_workers: int,
    layout: StoredWindowLayout | list[StoredWindowLayout],
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
    worker_pool = _create_preprocess_pool(num_workers) if num_workers > 1 else None
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
                num_workers,
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
    num_workers: Optional[int],
    layout: StoredWindowLayout | list[StoredWindowLayout],
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
    num_workers = num_workers or multiprocessing.cpu_count()
    sequence_count = len(_sequence_run_bounds(data)[0])
    maximum_batches = data.height if allow_sequence_splitting else sequence_count
    num_workers = min(num_workers, maximum_batches)
    if (
        not normalize_on_all_data
        and allow_sequence_splitting
        and split_method == "within_sequence"
        and num_workers > sequence_count
    ):
        raise ValueError(
            "Unsafe preprocessing configuration: normalize_on_all_data=False, "
            "allow_sequence_splitting=True, and split_method='within_sequence' "
            "would split sequences across worker batches "
            f"(effective num_workers={num_workers}, sequences={sequence_count}). This "
            "would make split-0 metadata fitting disagree with the materialized "
            "training split. Set allow_sequence_splitting=False, reduce num_workers "
            f"to at most {sequence_count}, or set normalize_on_all_data=True."
        )
    batch_limits = get_batch_limits(data, num_workers, allow_sequence_splitting)
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
        self.schemas = (
            schema if isinstance(schema, list) else [schema] * len(split_paths)
        )
        self.paths = [
            insert_top_folder(
                split_path.replace(f".{write_format}", f"-{process_id}.{write_format}"),
                target_dir,
            )
            for split_path in split_paths
        ]
        self._writers: dict[int, Any] = {}
        for group, path in enumerate(self.paths):
            if write_format == "parquet":
                self._writers[group] = pq.ParquetWriter(
                    path,
                    schema=pl.DataFrame(schema=self.schemas[group]).to_arrow().schema,
                    compression="snappy",
                )
            elif write_format == "csv":
                file = open(path, "wb")
                pl.DataFrame(schema=self.schemas[group]).write_csv(
                    file, include_header=True
                )
                self._writers[group] = file
            else:
                raise ValueError(
                    f"Merged output does not support write format {write_format!r}"
                )

    def append(self, group: int, windows: list[SequenceWindows]) -> None:
        if not windows:
            return
        data = _sequence_windows_list_to_long_dataframe(windows, self.schemas[group])
        if self.write_format == "parquet":
            self._writers[group].write_table(data.to_arrow())
        else:
            data.write_csv(self._writers[group], include_header=False)

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()


def _layout_for_split(
    layout: StoredWindowLayout | list[StoredWindowLayout], index: int
) -> StoredWindowLayout:
    return layout[index] if isinstance(layout, list) else layout


def _schema_for_split(schema: Any, width: int) -> Any:
    if not isinstance(schema, dict):
        return schema
    position_type = next(
        (dtype for key, dtype in schema.items() if key.isdigit()), None
    )
    return {
        **{key: dtype for key, dtype in schema.items() if not key.isdigit()},
        **{str(i): position_type for i in range(width - 1, -1, -1)},
    }


def _place_sequence_windows_for_splits(
    batch_arrays: BatchArrays,
    run_start: int,
    run_stop: int,
    sequence_id: int,
    layout: StoredWindowLayout | list[StoredWindowLayout],
    window_stride: int,
    split_ratios: Optional[list[float]],
    split_method: str,
    seed: int,
    sequence_split_assignments: Optional[dict[int, int]] = None,
    cumulative_split_ratios: Optional[np.ndarray] = None,
    alignment: Optional[PredictionAlignment] = None,
    split_values: Optional[list[Any]] = None,
) -> dict[int, tuple[WindowPlacement, np.ndarray | None]]:
    """Determine split membership, then place windows within each target interval."""
    length = run_stop - run_start
    if split_method == "between_sequence":
        assert split_ratios is not None
        split = (sequence_split_assignments or {}).get(sequence_id)
        if split is None:
            split = assign_sequence_to_split(sequence_id, split_ratios, seed)
        intervals = [(split, 0, length)]
    else:
        if split_method == "within_sequence":
            assert split_ratios is not None
            ratios = (
                cumulative_split_ratios
                if cumulative_split_ratios is not None
                else np.cumsum(split_ratios)
            )
            bounds = [0, *(ratios * length).astype(int).tolist()]
        elif split_method == "value_cutoff":
            cutoffs = _normalized_split_cutoffs(split_values)
            if cutoffs is None or batch_arrays.split_values is None:
                raise ValueError("value_cutoff splitting requires split_values")
            values = batch_arrays.split_values[run_start:run_stop]
            if np.any(values[1:] < values[:-1]):
                raise ValueError(
                    "split_column values must be non-decreasing within each sequence"
                )
            bounds = [
                0,
                *np.searchsorted(values, cutoffs, side="left").tolist(),
                length,
            ]
        else:
            raise ValueError(
                "split_method must be one of 'within_sequence', 'between_sequence', 'value_cutoff'"
            )
        intervals = [
            (i, low, high)
            for i, (low, high) in enumerate(zip(bounds, bounds[1:]))
            if high > low
        ]

    first = int(batch_arrays.item_positions[run_start])
    curriculum = {
        name: values[run_start:run_stop]
        for name, values in batch_arrays.curriculum_values.items()
    }
    placements = {}
    for split, low, high in intervals:
        placement = WindowPlacement.for_split(
            first,
            first + low,
            first + high,
            _layout_for_split(layout, split).window_length,
            window_stride,
            alignment if alignment and split in alignment.splits else None,
        )
        placements[split] = (
            placement,
            placement.curriculum(curriculum, first, sequence_id),
        )
    return placements


@beartype
def preprocess_batch(
    project_root: str,
    data_name_root: str,
    process_id: int,
    batch: pl.DataFrame,
    schema: Any,
    split_paths: list[str],
    layout: StoredWindowLayout | list[StoredWindowLayout],
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
    split_schemas = [
        _schema_for_split(schema, _layout_for_split(layout, i).window_length)
        for i in range(len(split_paths))
    ]
    sequence_count = len(batch_arrays.run_starts)
    cumulative_split_ratios = (
        np.cumsum(split_ratios) if split_ratios is not None else None
    )

    def extract_for_run(
        run_start: int, run_stop: int
    ) -> dict[int, Optional[SequenceWindows]]:
        sequence_id = int(batch_arrays.sequence_ids[run_start])
        placements = _place_sequence_windows_for_splits(
            batch_arrays,
            run_start,
            run_stop,
            sequence_id,
            layout,
            window_stride,
            split_ratios,
            split_method,
            seed,
            sequence_split_assignments,
            cumulative_split_ratios,
            alignment,
            split_values,
        )
        return {
            split: _dense_windows(
                batch_arrays, run_start, run_stop, placement, samples, data_columns
            )
            for split, (placement, samples) in placements.items()
        }

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
                    split_schemas[group],
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
        split_schemas,
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


def _dense_windows(batch, start, stop, placement, samples, columns):
    first = int(batch.item_positions[start])
    count = len(placement.starts)
    return SequenceWindows(
        sequence_id=int(batch.sequence_ids[start]),
        subsequence_ids=np.arange(count, dtype=np.int64),
        start_item_positions=placement.starts,
        left_pad_lengths=placement.padding,
        split_start_item_positions=np.full(count, placement.low, dtype=np.int64),
        split_end_item_positions=np.full(count, placement.high, dtype=np.int64),
        values={
            column: placement.gather(batch.values[column][start:stop], first)
            for column in columns
        },
        curriculum_columns=tuple(batch.curriculum_values),
        sample_positions=samples,
    )


@beartype
def extract_sequence_windows(
    data: pl.DataFrame,
    layout: StoredWindowLayout,
    stride_for_split: int,
    columns: list[str],
) -> Optional[SequenceWindows]:
    """Compatibility wrapper for extracting one sequence DataFrame."""
    if data.is_empty():
        return None
    batch = _batch_to_arrays(data, columns)
    first = int(batch.item_positions[0])
    placement = WindowPlacement.for_split(
        first, first, first + data.height, layout.window_length, stride_for_split
    )
    samples = placement.curriculum(
        batch.curriculum_values, first, int(batch.sequence_ids[0])
    )
    return _dense_windows(batch, 0, data.height, placement, samples, columns)


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
