import json
import math
import multiprocessing
import os
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union, cast

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from loguru import logger

from sequifier.artifacts.dependencies import DependencyManifest
from sequifier.config.metadata import (
    dataset_tensor_kind,
    dataset_tensor_version,
    validate_persisted_metadata_versions,
)
from sequifier.config.preprocess_config import (
    CardinalityLimitModel,
    PreprocessorModel,
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
    read_data,
    resolve_unified_polars_numeric_dtype,
)
from sequifier.io.atomic import write_json_atomic
from sequifier.io.feature_fitting import (
    FeatureStatistics,
    FittedColumns,
    FittingRules,
    cast_input_columns,
    configured_column_types,
    validate_cardinality_values,
)
from sequifier.io.images import _time_ms, media_files, validate_sources
from sequifier.io.partitioning import (
    sequence_assignments,
    sequence_split_overrides,
    table_content_keys,
)
from sequifier.io.preprocessing_workspace import (
    WORKSPACE_DIR,
    indexed_preparation,
    preprocessing_workspace,
)
from sequifier.io.pt_payload import StoredTensorBatch, load_pt_payload, save_pt_payload
from sequifier.io.sample_order import (
    CURRICULUM_COLUMN_PREFIX,
    curriculum_columns,
    curriculum_storage_column,
    curriculum_storage_columns,
    source_curriculum_column,
)
from sequifier.io.sequence import (
    DOMAIN_OBSERVATIONS,
    DOMAIN_STATIC,
    OBSERVATION_MASK,
    with_domain_record,
)
from sequifier.io.window_placement import (
    SplitWindowPlan,
    WindowPlacement,
    cutoff_intervals,
    get_subsequence_starts,
    split_intervals,
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
    return {
        column: CardinalityLimitModel.model_validate(value).model_dump(
            exclude_none=True
        )
        for column, value in (config or {}).items()
    }


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


def _between_sequence_assignments(lengths, plan, seed, assignments=None):
    ratios = list(plan.ratios)
    if assignments is None:
        assignments = sequence_split_overrides(lengths, {}, ratios, seed)
    counts = [0] * len(ratios)
    for sequence, length in lengths.items():
        split = assignments.get(sequence)
        if split is None:
            split = assign_sequence_to_split(sequence, ratios, seed)
        counts[split] += length
    plan.validate_training_share(counts)
    return assignments


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


def _is_temporal_dtype_name(dtype_name: str) -> bool:
    return dtype_name.startswith(("Date", "Datetime", "Time", "Duration"))


def _torch_dtype_for_column(dtype_name: str) -> torch.dtype:
    if _is_temporal_dtype_name(dtype_name):
        return torch.int64
    return PANDAS_TO_TORCH_TYPES[dtype_name]


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
        Preprocessor(config).run()
    logger.info("--- Preprocessing Complete ---")


@beartype
def _folder_input_files(
    data_path: str,
    read_format: str,
    exclude_path: str | None = None,
    *,
    exclude_directory: Path | None = None,
) -> list[str]:
    """Return folder input files in a stable traversal order."""
    file_paths = []
    for root, directories, files in os.walk(data_path):
        directories[:] = sorted(
            name
            for name in directories
            if name != WORKSPACE_DIR
            and (
                exclude_directory is None
                or (Path(root) / name).resolve() != exclude_directory
            )
        )
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
    rules: FittingRules
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
    features: FeatureStatistics


@dataclass(frozen=True)
class WindowWritePlan:
    config: PreprocessorModel
    name: str
    split_paths: list[str]
    target_dir: str
    schema: dict
    windows: SplitWindowPlan
    columns: list[str]
    col_types: dict[str, str]
    assignments: dict[int, int] | None

    @property
    def intermediate_paths(self):
        return [insert_top_folder(path, self.target_dir) for path in self.split_paths]

    def named(self, name):
        paths = [
            str(Path(path).with_name(Path(path).name.replace(self.name, name, 1)))
            for path in self.split_paths
        ]
        return replace(self, name=name, split_paths=paths)


@dataclass(frozen=True)
class _FolderFit:
    files: list[str]
    features: FittedColumns
    sequence_summaries: dict[int, list[int]]
    cutoff_counts: list[int] | None

    @property
    def spans_files(self) -> bool:
        return any(summary[3] > 1 for summary in self.sequence_summaries.values())


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
    _validate_declared_column_roles(data_columns, job.rules.categorical, job.rules.real)
    configured_col_types = configured_column_types(job.column_data_types, data_columns)
    data = cast_input_columns(data, data_columns, configured_col_types)
    source_col_types = {col: str(data.schema[col]) for col in data_columns}
    for col in job.rules.precomputed:
        if col not in data_columns:
            raise ValueError(f"Precomputed column {col} not found in {job.path}")
    _validate_cardinality_columns(
        job.rules.cardinality, data_columns, job.rules.precomputed
    )
    validate_cardinality_values(data, job.rules.cardinality)
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
    features = FeatureStatistics()
    features.observe(fitting_data, data_columns, job.rules, mask_column=job.mask_column)
    return _FolderMetadataResult(
        data.height,
        data_columns,
        source_col_types,
        configured_col_types,
        bool(curriculum_storage_columns(data.columns)),
        file_summaries,
        cutoff_counts,
        features,
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
                    summary[0] + split_intervals(summary[2], split_ratios)[0][1]
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
    training_slices = []
    for start, stop in zip(starts, stops):
        _, end = split_intervals(int(stop - start), split_ratios)[0]
        if end:
            training_slices.append(data.slice(int(start), end))
    return pl.concat(training_slices) if training_slices else data.head(0)


class Preprocessor:
    """Stateful preprocessing pipeline for single-file or folder inputs."""

    def __init__(self, config: PreprocessorModel):
        self.config = config
        self.indexed_prediction = False
        self.sequence_identities = {}
        self.sequence_dependencies = DependencyManifest()
        self.sequence_manifest_path = None
        self.sequence_view = config.sequence_view
        self.external_static_source = None
        self.preparation = None
        self.static_cardinality = {}

    def run(self, *, indexed_writer=None, indexed_prediction=False) -> None:
        if (
            self.config.read_format == "sequence"
            or Path(self.config.preprocessing_data_path).suffix.lower() == ".json"
        ) and Path(self.config.preprocessing_data_path).is_file():
            from sequifier.trajectory.sequence_output import supervised_manifest_input

            with supervised_manifest_input(
                self.config.preprocessing_data_path,
                self.config.sequence_view,
                dependencies=self.sequence_dependencies,
                mask_column=self.config.mask_column,
                split_column=self.config.split_column,
                curriculum_columns=curriculum_columns(self.config.curriculum_column),
                indexed_passthrough=self.config.indexed_passthrough,
            ) as (
                table,
                manifest,
                static_path,
            ):
                columns = self.config.selected_columns
                if columns is None:
                    columns = manifest.get("observation_columns")
                config = self.config.model_copy(
                    update={
                        "preprocessing_data_path": str(table),
                        "read_format": "parquet",
                        "sequence_view": "decisions",
                        "mask_column": self.config.mask_column
                        or (
                            OBSERVATION_MASK
                            if manifest.get("masked_observations")
                            else None
                        ),
                        "selected_columns": columns,
                    }
                )
                if static_path and not config.static_data_path:
                    config.static_data_path = str(static_path)
                run = Preprocessor(config)
                run.sequence_view = self.config.sequence_view
                run.external_static_source = self.config.static_data_path
                run.sequence_dependencies = self.sequence_dependencies
                if run.external_static_source:
                    run.sequence_dependencies.add(run.external_static_source)
                run.sequence_manifest_path = Path(
                    self.config.preprocessing_data_path
                ).resolve()
                run._run_table(self.config, indexed_writer, indexed_prediction)
            return
        if self.config.sequence_view == "observations":
            raise ValueError(
                "sequence_view observations requires a sequence manifest or trajectory"
            )
        self._run_table(self.config, indexed_writer, indexed_prediction)

    def table_input_paths(self):
        config = self.config
        source = Path(config.preprocessing_data_path)
        files = (
            [source]
            if source.is_file()
            else [
                Path(path)
                for path in _folder_input_files(
                    str(source),
                    config.read_format,
                    config.static_data_path,
                    # A published store may sit under its raw input folder.
                    exclude_directory=(
                        Path(config.project_root) / config.indexed_output_path
                    ).resolve()
                    if config.indexed_output_path
                    else None,
                )
            ]
        )
        if config.static_data_path:
            files.append(Path(config.static_data_path))
        return files

    def _run_table(self, authored, indexed_writer, indexed_prediction):
        from sequifier.io.indexed_sequences import IndexedSequenceWriter

        if authored.indexed_output_path is None or indexed_writer is not None:
            self._process(indexed_writer, indexed_prediction)
            return
        root = Path(authored.project_root).resolve()
        destination = root / authored.indexed_output_path
        with preprocessing_workspace(
            destination, indexed=True, resume=not authored.reprocess
        ) as work:
            dependencies = DependencyManifest()
            if self.sequence_manifest_path:
                dependencies.merge(self.sequence_dependencies)
            else:
                for path in self.table_input_paths():
                    dependencies.add(path)
            if authored.metadata_config_path:
                dependencies.add(root / authored.metadata_config_path)
            for path in (root / "configs/id_maps").glob("*.json"):
                dependencies.add(path)
            locations = [source.path for source in authored.image_sources.values()]
            if prefix := authored.preconditioning_image:
                locations.extend((prefix.paths or {}).values())
            for location in filter(None, locations):
                dependencies.add_media_directory(
                    root / location, media_files(location, authored.project_root)
                )
            effective, self.preparation = indexed_preparation(
                authored, destination, work, dependencies.identity
            )
            if work is not None:
                self.config = self.config.model_copy(update={"run": effective.run})
                self.sequence_dependencies = dependencies
                with IndexedSequenceWriter(destination, workspace=work) as writer:
                    self._process(writer, indexed_prediction)

    def source_provenance(self):
        if self.sequence_manifest_path:
            source = {
                "manifest_path": str(self.sequence_manifest_path),
                "sequence_view": self.sequence_view,
            }
        else:
            config = self.config
            source = {
                "preprocessing_data_path": str(
                    Path(config.preprocessing_data_path).absolute()
                ),
                "static_data_path": str(Path(config.static_data_path).absolute())
                if config.static_data_path
                else None,
                "indexed_passthrough": config.indexed_passthrough,
            }
            if self.preparation is None:
                # Fresh inference supplies its own writer and skips the restart workspace.
                for path in self.table_input_paths():
                    self.sequence_dependencies.add(path)
        self.sequence_dependencies.revalidate()
        return {
            "source": source,
            "files": [
                {"path": location, "sha256": checksum}
                for location, checksum in sorted(
                    self.sequence_dependencies.identity.items()
                )
            ],
            **(
                {"preparation": self.preparation}
                if self.preparation is not None
                else {}
            ),
        }

    def _read_sequence_identities(self):
        from sequifier.io.sequence import episode_identity
        from sequifier.io.values import domain_scalar

        config = self.config
        path = Path(config.preprocessing_data_path)
        files = (
            [path]
            if path.is_file()
            else _folder_input_files(
                str(path), config.read_format, config.static_data_path
            )
        )
        reader = pl.scan_parquet if config.read_format == "parquet" else pl.scan_csv
        sources = []
        for file in files:
            scan = reader(file)
            schema = scan.collect_schema()
            fields = [name for name in ("sequenceName", "__group") if name in schema]
            if fields and config.read_format == "csv":
                scan = pl.scan_csv(file, infer_schema_length=None)
            sources.append((scan, fields))
        if not any(fields for _, fields in sources):
            return
        owners = {}
        for scan, fields in sources:
            for row in (
                scan.select("sequenceId", *fields)
                .unique()
                .collect()
                .iter_rows(named=True)
            ):
                key = row["sequenceId"]
                label = row.get("sequenceName")
                group = row.get("__group")
                identity = episode_identity(key, label)
                if owners.setdefault(identity, key) != key:
                    raise ValueError(f"duplicate episode ID {identity!r}")
                value = (
                    identity if label is not None else None,
                    str(domain_scalar(group, "collection group"))
                    if group is not None
                    else None,
                )
                if self.sequence_identities.setdefault(key, value) != value:
                    raise ValueError(
                        f"Episode identity or group changes within sequence {key}"
                    )

        for key, (label, group) in self.sequence_identities.items():
            if label is None:
                self.sequence_identities[key] = (episode_identity(key), group)

    def _sequence_assignments(self, sequences):
        config = self.config
        fractions = _require_split_ratios(config.split_ratios)
        content = {}
        if len(fractions) > 1:
            path = Path(config.preprocessing_data_path)
            files = (
                [str(path)]
                if path.is_file()
                else _folder_input_files(
                    str(path), config.read_format, config.static_data_path
                )
            )
            scans = [
                pl.scan_parquet(name)
                if config.read_format == "parquet"
                else pl.scan_csv(name, infer_schema_length=None)
                for name in files
            ]
            table = pl.concat(scans, how="diagonal_relaxed")
            if config.max_rows is not None:
                if config.depth_layouts:
                    coords = ["sequenceId", "itemPosition"]
                    selected = (
                        table.select(coords).unique().sort(coords).head(config.max_rows)
                    )
                    table = table.join(selected, on=coords, how="semi")
                else:
                    table = table.head(config.max_rows)
            content = table_content_keys(table)
            if self._static_source is not None:
                static = table_content_keys(self._static_source.lazy())
                content = {seq: (key, static[seq]) for seq, key in content.items()}
        return sequence_assignments(
            sequences,
            {
                seq: group
                for seq, (_, group) in self.sequence_identities.items()
                if seq in sequences
            },
            {seq: content[seq] for seq in sequences if seq in content},
            fractions,
            config.seed,
        )

    def _process(self, indexed_writer, indexed_prediction) -> None:
        from sequifier.io.preprocessing_output import IndexedOutput, WindowOutput

        config = self.config
        self.indexed_prediction = indexed_prediction
        self.image_time_rows: list[tuple[int, int, int]] = []
        self._frozen_metadata = None
        if config.metadata_config_path is not None:
            with open(Path(config.project_root) / config.metadata_config_path) as file:
                self._frozen_metadata = json.load(file)
            validate_persisted_metadata_versions(self._frozen_metadata)
            validate_special_token_ids(
                self._frozen_metadata["special_token_ids"],
                source=f"metadata config '{config.metadata_config_path}'",
            )

        self.data_name_root = Path(config.preprocessing_data_path).stem
        self.target_dir = f"{self.data_name_root}-temp"
        _normalized_split_cutoffs(config.split_values)
        self.n_splits = len(config.window_lengths)
        self.storage_layout = StoredWindowLayout(
            config.window_lengths[0],
            config.max_target_offset,
            ContractVersions.STORED_WINDOW_LAYOUT,
        )
        self.window_plan = SplitWindowPlan.from_settings(config)
        if config.split_method == "within_sequence" and config.split_ratios is not None:
            self.window_plan.validate_training_share(config.split_ratios)
        self.column_data_types = _normalize_column_types(config.column_data_types)
        self.output = (
            WindowOutput(self)
            if indexed_writer is None
            else IndexedOutput(self, indexed_writer)
        )
        frozen_metadata = self._frozen_metadata if self.output.indexed else None
        self.normalize_real_columns = config.normalize_real_columns
        if frozen_metadata is not None:
            frozen_types = _column_types_from_metadata(frozen_metadata) or {}
            if any(
                frozen_types.get(column) != dtype
                for column, dtype in (self.column_data_types or {}).items()
            ):
                raise ValueError(
                    "Indexed output types must match the original metadata"
                )
            self.normalize_real_columns = frozen_metadata.get(
                "normalize_real_columns", True
            )
        self.metadata_fitted_on_all_data = config.normalize_on_all_data
        np.random.seed(config.seed)
        self.num_workers = config.num_workers or multiprocessing.cpu_count()
        self.has_sample_positions = False
        self.output_features: FittedColumns
        self.image_manifest = self._validate_image_sources()
        self._setup_directories()

        self._read_sequence_identities()
        selected_columns = config.selected_columns
        if (
            self.sequence_identities
            or config.time_alignment_column is not None
            or config.indexed_passthrough
        ):
            if selected_columns is None:
                source_file = (
                    config.preprocessing_data_path
                    if os.path.isfile(config.preprocessing_data_path)
                    else _folder_input_files(
                        config.preprocessing_data_path,
                        config.read_format,
                        config.static_data_path,
                    )[0]
                )
                source_columns = (
                    pq.read_schema(source_file).names
                    if config.read_format == "parquet"
                    else pl.scan_csv(source_file).collect_schema().names()
                )
                selected_columns = [
                    column
                    for column in source_columns
                    if column
                    not in {
                        *INPUT_METADATA_COLUMNS,
                        "sequenceName",
                        "__group",
                        config.time_alignment_column,
                        config.mask_column,
                        *config.indexed_passthrough,
                    }
                ]
            else:
                selected_columns = [
                    column
                    for column in selected_columns
                    if column != config.time_alignment_column
                ]

        if selected_columns is not None:
            selected_columns = ["sequenceId", "itemPosition"] + selected_columns
            if (
                config.mask_column is not None
                and config.mask_column in selected_columns
            ):
                raise ValueError(
                    f"'{config.mask_column}' is not allowed to be in 'selected_columns'"
                )

        self._setup_split_paths(config.write_format, self.n_splits)
        self._static_source = self._load_static_source()
        cardinality = _normalize_cardinality_config(config.cardinality_config)
        static_names = (
            set(self._static_source.columns) - {"sequenceId"}
            if self._static_source is not None
            else set()
        )
        if self._frozen_metadata is not None:
            persisted = _normalize_cardinality_config(
                self._frozen_metadata.get("cardinality_config")
            )
            if cardinality and cardinality != persisted:
                raise ValueError("cardinality_config must match metadata_config_path")
            cardinality = persisted
        self.static_cardinality = {
            name: rule for name, rule in cardinality.items() if name in static_names
        }
        frozen_static = set((self._frozen_metadata or {}).get("static_columns", ()))
        if frozen_metadata is not None and (unexpected := static_names - frozen_static):
            raise ValueError(f"Unexpected static columns: {sorted(unexpected)}")
        self.cardinality_config = {
            name: rule
            for name, rule in cardinality.items()
            if name not in static_names and name not in frozen_static
        }
        if config.depth_layouts:
            from sequifier.io.depth_preprocess import preprocess_depth

            preprocess_depth(self, selected_columns)
            self.output.finish()
            return

        if (
            isinstance(self.output, WindowOutput)
            and self.output.cleanup_checkpoint.is_file()
        ):
            self._restore_resume_manifest_state(selected_columns, config.write_format)
            self.output.finish()
            return

        if os.path.isfile(config.preprocessing_data_path):
            self._process_file(selected_columns)
        else:
            self._process_folder(selected_columns)
        self.output.finish()

    def _write_feature_metadata(self, selected_columns, fitted):
        self._write_or_validate_resume_manifest(
            selected_columns, self.config.write_format, fitted
        )
        self._export_metadata(fitted)

    def _capture_source(self, data, columns, part):
        captured = with_domain_record(
            data, columns, DOMAIN_OBSERVATIONS, self.config.mask_column
        )
        return self.output.capture_source(data, captured, part)

    def _process_file(self, selected_columns):
        config = self.config
        data = _load_and_preprocess_data(
            config.preprocessing_data_path,
            config.read_format,
            selected_columns,
            config.max_rows,
            config.mask_column,
            config.curriculum_column,
            split_column=config.split_column,
            split_values=config.split_values,
        )
        self.has_sample_positions = bool(curriculum_storage_columns(data.columns))
        data_columns = _get_data_columns(data, config.mask_column)
        _validate_declared_column_roles(
            data_columns, config.categorical_columns, config.real_columns
        )
        configured_col_types = configured_column_types(
            self.column_data_types, data_columns
        )
        source_rows = data
        data = cast_input_columns(data, data_columns, configured_col_types)
        sequence_split_assignments = None
        if config.split_method == "between_sequence":
            lengths = dict(data.group_by("sequenceId").agg(pl.len()).iter_rows())
            sequence_split_assignments = self._sequence_assignments(lengths)
            sequence_split_assignments = _between_sequence_assignments(
                lengths, self.window_plan, config.seed, sequence_split_assignments
            )
        if config.split_method == "value_cutoff":
            cutoffs = _normalized_split_cutoffs(config.split_values)
            assert cutoffs is not None
            self.window_plan.validate_training_share(
                _cutoff_row_counts(data.get_column(SPLIT_VALUE_COLUMN), cutoffs)
            )
        if self._frozen_metadata is not None:
            fitted = self._frozen_temporal_features(
                data_columns=data_columns, schema=data.schema
            )
        else:
            precomputed_id_maps = load_precomputed_id_maps(
                config.project_root, data_columns, config.use_precomputed_maps
            )
            _validate_cardinality_columns(
                self.cardinality_config, data_columns, precomputed_id_maps
            )
            validate_cardinality_values(data, self.cardinality_config)

            fitting_data = (
                data
                if config.normalize_on_all_data
                else _fit_data_for_split_zero(
                    data,
                    config.split_ratios,
                    config.split_method,
                    config.seed,
                    sequence_split_assignments,
                    split_values=config.split_values,
                )
            )
            rules = FittingRules(
                precomputed_id_maps,
                self.cardinality_config,
                config.categorical_columns,
                config.real_columns,
            )
            features = FeatureStatistics()
            features.observe(
                fitting_data, data_columns, rules, mask_column=config.mask_column
            )
            id_maps, selected_columns_statistics = features.finish(rules)
            fitted = FittedColumns.resolve(
                data_columns,
                id_maps,
                selected_columns_statistics,
                data.schema,
                types=configured_col_types,
            )
        self._write_feature_metadata(selected_columns, fitted)
        data = fitted.transform(
            self._capture_source(source_rows, data_columns, "items"),
            self.normalize_real_columns,
            self.cardinality_config,
            config.mask_column,
            input_types=configured_col_types,
        )

        data = self._add_indexed_passthrough(data, config.preprocessing_data_path)
        self.output.write_sequences(data, sequence_split_assignments)

    def _process_folder(self, selected_columns):
        config = self.config
        files_to_process = _folder_input_files(
            config.preprocessing_data_path,
            config.read_format,
            config.static_data_path if config.static_data_path else None,
        )
        self.has_sample_positions = _input_has_sample_positions(
            files_to_process, config.read_format, config.curriculum_column
        )
        folder_sequence_summaries = None
        folder_cutoff_counts = None
        combine_audit_and_fit = (
            config.metadata_config_path is None
            and not config.normalize_on_all_data
            and config.split_method == "within_sequence"
        )
        if not combine_audit_and_fit and (
            config.metadata_config_path
            or not config.normalize_on_all_data
            or self.sequence_identities
        ):
            (
                files_to_process,
                folder_sequence_summaries,
                folder_cutoff_counts,
            ) = _audit_folder_sequence_summaries(
                files_to_process,
                config.read_format,
                config.max_rows,
                config.split_column if config.metadata_config_path else None,
                config.split_values if config.metadata_config_path else None,
                self.num_workers,
            )
        sequence_split_assignments = (
            self._sequence_assignments(folder_sequence_summaries)
            if folder_sequence_summaries is not None
            and config.split_method == "between_sequence"
            else None
        )

        if self._frozen_metadata is not None:
            features = self._frozen_temporal_features(selected_columns=selected_columns)
            assert folder_sequence_summaries is not None
            _validate_declared_column_roles(
                features.data_columns, config.categorical_columns, config.real_columns
            )
            fit = _FolderFit(
                files=files_to_process,
                features=features,
                sequence_summaries=folder_sequence_summaries,
                cutoff_counts=folder_cutoff_counts,
            )
        else:
            fit = self._get_column_metadata_across_files(
                selected_columns,
                files_to_process,
                sequence_split_assignments,
                folder_sequence_summaries,
            )
            if combine_audit_and_fit and fit.spans_files:
                logger.info(
                    "Sequences span files; refitting metadata with global split boundaries."
                )
                fit = self._get_column_metadata_across_files(
                    selected_columns, fit.files, None, fit.sequence_summaries
                )

        if config.split_method == "between_sequence":
            lengths = {
                sid: summary[2] for sid, summary in fit.sequence_summaries.items()
            }
            if sequence_split_assignments is None:
                sequence_split_assignments = self._sequence_assignments(lengths)
            sequence_split_assignments = _between_sequence_assignments(
                lengths, self.window_plan, config.seed, sequence_split_assignments
            )
        if config.split_method == "value_cutoff":
            assert fit.cutoff_counts is not None
            self.window_plan.validate_training_share(fit.cutoff_counts)

        self._write_feature_metadata(selected_columns, fit.features)
        if self.output.indexed is True or fit.spans_files:
            if fit.spans_files:
                logger.warning(
                    "Sequences span multiple input files; processing the folder as "
                    "one logical dataset to preserve sequence boundaries."
                )
            self._process_fragmented_folder(
                fit, selected_columns, sequence_split_assignments
            )
        else:
            self._process_batches_multiple_files(
                fit,
                selected_columns,
                self.output.worker_plan(sequence_split_assignments),
            )

    def _add_indexed_passthrough(self, data, path):
        if not self.config.indexed_passthrough:
            return data
        coordinates = ["sequenceId", "itemPosition"]
        auxiliary = read_data(
            path,
            self.config.read_format,
            columns=coordinates + self.config.indexed_passthrough,
        ).with_columns(pl.col(coordinates).cast(pl.Int64))
        return data.join(
            auxiliary, on=coordinates, how="left", validate="1:1", maintain_order="left"
        )

    def _validate_image_sources(self) -> dict[str, Any]:
        if not self.config.image_sources and self.config.preconditioning_image is None:
            return {"time_alignment_column": self.config.time_alignment_column}
        if self.config.depth_layouts:
            files = (
                [self.config.preprocessing_data_path]
                if os.path.isfile(self.config.preprocessing_data_path)
                else _folder_input_files(
                    self.config.preprocessing_data_path,
                    self.config.read_format,
                    self.config.static_data_path,
                )
            )
            positions_by_sequence: dict[int, set[int]] = {}
            rows_read = 0
            for file in files:
                remaining = (
                    None
                    if self.config.max_rows is None
                    else self.config.max_rows - rows_read
                )
                if remaining is not None and remaining <= 0:
                    break
                coordinates = read_data(
                    file, self.config.read_format, columns=list(INPUT_METADATA_COLUMNS)
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
        elif os.path.isfile(self.config.preprocessing_data_path):
            _, summaries, _ = _audit_folder_file(
                (
                    self.config.preprocessing_data_path,
                    self.config.read_format,
                    self.config.max_rows,
                    None,
                    None,
                )
            )
        else:
            files = _folder_input_files(
                self.config.preprocessing_data_path,
                self.config.read_format,
                self.config.static_data_path,
            )
            _, summaries, _ = _audit_folder_sequence_summaries(
                files, self.config.read_format, self.config.max_rows, num_workers=1
            )
        if any(summary[0] != 0 for summary in summaries.values()):
            raise ValueError("Image sequences require itemPosition to start at zero")
        item_times = None
        time_mode = None
        time_index_path = None
        if self.config.time_alignment_column is not None:
            files = (
                [self.config.preprocessing_data_path]
                if os.path.isfile(self.config.preprocessing_data_path)
                else _folder_input_files(
                    self.config.preprocessing_data_path,
                    self.config.read_format,
                    self.config.static_data_path,
                )
            )
            rows_read = 0
            values_by_sequence: dict[int, dict[int, int]] = {}
            for file in files:
                remaining = (
                    None
                    if self.config.max_rows is None
                    else self.config.max_rows - rows_read
                )
                if remaining is not None and remaining <= 0:
                    break
                data = read_data(
                    file,
                    self.config.read_format,
                    columns=[
                        "sequenceId",
                        "itemPosition",
                        self.config.time_alignment_column,
                    ],
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
                        if (
                            self.config.depth_layouts
                            and positions[int(position)] == stamp
                        ):
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
            time_index_path = self.output.time_index_name
        manifest = validate_sources(
            self.config.image_sources,
            self.config.preconditioning_image,
            {sequence_id: summary[2] for sequence_id, summary in summaries.items()},
            self.config.project_root,
            item_times=item_times,
            time_mode=time_mode,
            time_index_path=time_index_path,
            video_index_dir=self.output.video_index_dir,
        )
        self.output.bind_media(manifest)
        manifest["time_alignment_column"] = self.config.time_alignment_column
        return manifest

    def _load_static_source(self) -> Optional[pl.DataFrame]:
        """Validate the one-to-one source table before writing any windows."""
        if self.config.static_data_path is None:
            return None
        path = self.config.static_data_path
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
        declared = set(
            self.config.static_categorical_columns + self.config.static_real_columns
        )
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
            [self.config.preprocessing_data_path]
            if os.path.isfile(self.config.preprocessing_data_path)
            else _folder_input_files(
                self.config.preprocessing_data_path,
                self.config.read_format,
                self.config.static_data_path if self.config.static_data_path else None,
            )
        )
        sequence_ids: set[int] = set()
        for source in source_files:
            scan = (
                pl.scan_parquet(source)
                if self.config.read_format == "parquet"
                else pl.scan_csv(source)
            )
            source_schema = scan.collect_schema()
            if not is_integer_dtype_name(str(source_schema["sequenceId"])):
                raise ValueError(
                    f"Sequence data {source!r} must use integer sequenceId values"
                )
            overlap = set(columns) & set(source_schema.names())
            if self.sequence_manifest_path and not self.external_static_source:
                # The manifest projection repeats these static fields for window
                # placement. They still have a single feature in the static table.
                placement_columns = {
                    self.config.split_column,
                    *curriculum_columns(self.config.curriculum_column),
                }
                overlap -= placement_columns.difference(
                    self.config.selected_columns or ()
                )
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

    def _finalize_static_source(self) -> None:
        """Fit static transforms once per sequence and persist one feature table."""
        assert self._static_source is not None
        static = self._static_source
        columns = [column for column in static.columns if column != "sequenceId"]
        existing = self._frozen_metadata
        if existing is not None and set(columns) <= set(
            existing.get("static_columns", [])
        ):
            fitted = FittedColumns.from_metadata(
                columns,
                existing,
                categorical=self.config.static_categorical_columns,
                real=self.config.static_real_columns,
            )
        else:
            fitting = (
                static
                if self.config.normalize_on_all_data
                else static.filter(
                    pl.col("sequenceId").is_in(self.output.training_sequence_ids())
                )
            )
            rules = FittingRules(
                cardinality=self.static_cardinality,
                categorical=self.config.static_categorical_columns,
                real=self.config.static_real_columns,
            )
            features = FeatureStatistics()
            features.observe(fitting, columns, rules)
            maps, stats = features.finish(rules)
            fitted = FittedColumns.resolve(columns, maps, stats, static.schema)
        static = with_domain_record(static, columns, DOMAIN_STATIC)
        static = fitted.transform(
            static, self.normalize_real_columns, self.static_cardinality
        )
        self.output.write_static(static, fitted.metadata())

    def _frozen_temporal_features(
        self, *, data_columns=None, selected_columns=None, schema=None
    ):
        metadata = self._frozen_metadata
        assert metadata is not None
        types = _column_types_from_metadata(metadata)
        static = set(metadata.get("static_columns", ()))
        if data_columns is None:
            if types is None:
                raise ValueError(
                    "Folder preprocessing requires frozen column_data_types"
                )
            excluded = static.union(_reserved_input_columns(self.config.mask_column))
            data_columns = [
                column
                for column in types
                if column not in excluded
                and (not self.indexed_prediction or column in (selected_columns or ()))
            ]
        elif overlap := static.intersection(data_columns):
            raise ValueError(
                f"Temporal inputs include frozen static columns: {sorted(overlap)}"
            )
        fitted = FittedColumns.from_metadata(
            data_columns,
            metadata,
            types=types,
            schema=schema,
            categorical=self.config.categorical_columns,
            real=self.config.real_columns,
        )
        self.metadata_fitted_on_all_data = metadata.get("normalize_on_all_data", True)
        if self.indexed_prediction:
            self.cardinality_config = {
                name: rule
                for name, rule in self.cardinality_config.items()
                if name in data_columns
            }
        _validate_cardinality_columns(self.cardinality_config, data_columns, {})
        configured = configured_column_types(self.column_data_types, data_columns)
        if self.config.depth_layouts and configured and configured != fitted.col_types:
            raise ValueError("Configured output types differ from reused metadata")
        return replace(fitted, col_types=configured) if configured else fitted

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
                    for column in curriculum_columns(self.config.curriculum_column)
                }
            )
        schema["inputCol"] = pl.String

        if self.config.write_format == "parquet":
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
        selected_columns: Optional[list[str]],
        files_to_process: list[str],
        sequence_split_assignments: Optional[dict[int, int]],
        sequence_summaries: Optional[dict[int, list[int]]],
    ) -> _FolderFit:
        """Fit folder metadata with ordered reduction of per-file summaries."""
        max_rows = self.config.max_rows
        column_data_types = self.column_data_types

        precomputed_id_maps = load_precomputed_id_maps(
            self.config.project_root, None, self.config.use_precomputed_maps
        )
        rules = FittingRules(
            precomputed_id_maps,
            self.cardinality_config,
            self.config.categorical_columns,
            self.config.real_columns,
        )
        features = FeatureStatistics()
        col_types: Optional[dict[str, str]] = None
        data_columns: Optional[list[str]] = None
        sample_position_presence: Optional[bool] = None
        selected_file_paths: list[str] = []
        coordinate_summaries: dict[int, list[int]] = {}
        cutoffs = (
            _normalized_split_cutoffs(self.config.split_values)
            if self.config.split_method == "value_cutoff"
            else None
        )
        cutoff_counts = [0] * (len(cutoffs) + 1) if cutoffs is not None else None
        rows_read = 0

        def job_for(path: str, row_limit: Optional[int]) -> _FolderMetadataJob:
            return _FolderMetadataJob(
                path=path,
                read_format=self.config.read_format,
                selected_columns=selected_columns,
                row_limit=row_limit,
                mask_column=self.config.mask_column,
                curriculum_column=self.config.curriculum_column,
                split_column=self.config.split_column,
                split_values=self.config.split_values,
                column_data_types=column_data_types,
                rules=rules,
                normalize_on_all_data=self.config.normalize_on_all_data,
                split_ratios=self.config.split_ratios,
                split_method=self.config.split_method,
                seed=self.config.seed,
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
            features.merge(result.features)

        logger.info(f"Data path: {self.config.preprocessing_data_path}")
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
        id_maps, selected_columns_statistics = features.finish(rules)
        if data_columns is None:
            raise RuntimeError("data_columns was not initialized correctly.")
        if col_types is None:
            raise RuntimeError("col_types was not initialized correctly.")
        self.has_sample_positions = bool(sample_position_presence)
        return _FolderFit(
            files=selected_file_paths,
            features=FittedColumns.resolve(
                data_columns,
                id_maps,
                selected_columns_statistics,
                col_types,
                types=col_types if column_data_types is not None else None,
            ),
            sequence_summaries=coordinate_summaries,
            cutoff_counts=cutoff_counts,
        )

    @beartype
    def _setup_directories(self) -> None:
        """Prepare or validate preprocessing temp directories."""

        temp_path = os.path.join(self.config.project_root, "data", self.target_dir)

        if (self.config.reprocess or self.output.indexed) and os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.makedirs(temp_path, exist_ok=True)

    @beartype
    def _setup_split_paths(self, write_format: str, n_splits: int) -> None:
        """Set final split output paths."""
        split_paths = [
            os.path.join(
                self.config.project_root,
                "data",
                f"{self.data_name_root}-split{i}.{write_format}",
            )
            for i in range(n_splits)
        ]

        self.split_paths = split_paths

    @beartype
    def _process_fragmented_folder(
        self,
        fit: _FolderFit,
        selected_columns: Optional[list[str]],
        sequence_split_assignments: Optional[dict[int, int]],
    ) -> None:
        """Process cross-file fragments through bounded disk-backed hash buckets."""
        total_input_bytes = sum(os.path.getsize(path) for path in fit.files)
        target_bucket_bytes = 256 * 1024 * 1024
        bucket_count = min(
            4096,
            max(self.num_workers, math.ceil(total_input_bytes / target_bucket_bytes)),
        )
        bucket_dir = os.path.join(
            self.config.project_root, "data", self.target_dir, "fragment-buckets"
        )
        os.makedirs(bucket_dir, exist_ok=True)
        bucket_files: dict[int, list[str]] = {}
        rows_read = 0

        for file_index, path in enumerate(fit.files):
            remaining_rows = (
                None
                if self.config.max_rows is None
                else self.config.max_rows - rows_read
            )
            if remaining_rows is not None and remaining_rows <= 0:
                break
            data = _load_and_preprocess_data(
                path,
                self.config.read_format,
                selected_columns,
                remaining_rows,
                self.config.mask_column,
                self.config.curriculum_column,
                False,
                self.config.split_column,
                self.config.split_values,
            )
            data = self._capture_source(
                data, fit.features.data_columns, str(file_index)
            )
            data = fit.features.transform(
                data,
                self.normalize_real_columns,
                self.cardinality_config,
                self.config.mask_column,
                input_types=fit.features.col_types,
            )
            data = self._add_indexed_passthrough(data, path)
            data = data.with_columns(
                (
                    pl.col("sequenceId").hash(seed=self.config.seed % (2**64))
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

        worker_pool = (
            _create_preprocess_pool(self.num_workers)
            if self.num_workers > 1 and not self.output.indexed
            else None
        )
        try:
            for bucket, paths in sorted(bucket_files.items()):
                data = _validate_sequence_coordinates(
                    pl.concat([pl.read_parquet(path) for path in paths]),
                    f"fragment bucket {bucket}",
                )
                self.output.write_sequences(
                    data, sequence_split_assignments, bucket=bucket, pool=worker_pool
                )
        finally:
            if worker_pool is not None:
                worker_pool.close()
                worker_pool.join()
            shutil.rmtree(bucket_dir)

    @beartype
    def _process_batches_multiple_files(
        self,
        fit: _FolderFit,
        selected_columns: Optional[list[str]],
        plan: WindowWritePlan,
    ) -> None:
        """Dispatch folder preprocessing by file or by process shard."""
        config = self.config
        process_by_file = config.process_by_file
        if not process_by_file and config.max_rows is not None:
            logger.warning(
                "process_by_file=False cannot preserve a folder-global max_rows "
                "limit across file shards; processing files serially instead."
            )
            process_by_file = True

        process_files = partial(
            _process_batches_multiple_files_inner,
            plan,
            fit.features,
            selected_columns,
            self.normalize_real_columns,
            self.cardinality_config,
            self.num_workers if process_by_file else 1,
        )
        file_paths = fit.files
        if process_by_file:
            results = [process_files(0, file_paths)]
        else:
            worker_count = min(self.num_workers, len(file_paths))
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

            with _create_preprocess_pool(worker_count) as pool:
                results = pool.starmap(process_files, enumerate(file_sets))

        if config.merge_output:
            merge_window_outputs(
                [paths for worker in results for paths in worker],
                plan.split_paths,
                config.write_format,
            )

    @beartype
    def _cleanup(self, write_format: str) -> None:
        """Move split outputs and write folder metadata."""

        logger.info("Start cleanup")
        temp_output_path = os.path.join(
            self.config.project_root, "data", self.target_dir
        )
        directory = Path(temp_output_path)

        if not self.config.merge_output:
            for i, split_path in enumerate(self.split_paths):
                split = f"split{i}"
                folder_path = os.path.join(
                    self.config.project_root, "data", f"{self.data_name_root}-{split}"
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
        tensor_kind = dataset_tensor_kind(self.config.depth_layouts)
        return {
            "depth_layouts": self.config.depth_layouts.model_dump(mode="json"),
            "tensor_payload_kind": tensor_kind,
            "tensor_payload_version": dataset_tensor_version(tensor_kind),
            **self.config.window_metadata(),
            "stored_window_layout_version": self.storage_layout.version,
        }

    @beartype
    def _write_or_validate_resume_manifest(
        self,
        selected_columns: Optional[list[str]],
        write_format: str,
        fitted: FittedColumns,
    ) -> None:
        manifest = {
            "manifest_version": ContractVersions.PREPROCESSING_MANIFEST,
            "preprocessing_config": {
                **self._layout_metadata(),
                "read_format": self.config.read_format,
                "static_data_path": self.external_static_source
                if self.sequence_manifest_path
                else self.config.static_data_path,
                **(
                    {
                        "sequence_manifest": self.source_provenance(),
                        "sequence_view": self.sequence_view,
                    }
                    if self.sequence_manifest_path
                    else {}
                ),
                "static_categorical_columns": self.config.static_categorical_columns,
                "static_real_columns": self.config.static_real_columns,
                **self.image_manifest,
                "write_format": write_format,
                "merge_output": self.config.merge_output,
                "selected_columns": selected_columns,
                "data_columns": fitted.data_columns,
                "categorical_columns": self.config.categorical_columns,
                "real_columns": self.config.real_columns,
                **self.config.split_metadata(),
                "seed": self.config.seed,
                "max_rows": self.config.max_rows,
                "process_by_file": self.config.process_by_file,
                "mask_column": self.config.mask_column,
                **(
                    {"curriculum_column": self.config.curriculum_column}
                    if self.config.curriculum_column is not None
                    else {}
                ),
                "use_precomputed_maps": self.config.use_precomputed_maps,
                "cardinality_config": self.cardinality_config | self.static_cardinality,
                **fitted.metadata(for_resume=True),
                "normalize_real_columns": self.normalize_real_columns,
                "normalize_on_all_data": self.config.normalize_on_all_data,
                "special_token_ids": SPECIAL_TOKEN_IDS.ids_by_label,
            },
        }
        scratch = Path(self.config.project_root) / "data" / self.target_dir
        manifest_path = scratch / "preprocess-manifest.json"
        # A worker or batch number identifies the same windows only under the
        # same partitioning settings.
        partitioning = dict(
            num_workers=self.config.num_workers or multiprocessing.cpu_count(),
            allow_sequence_splitting=self.config.allow_sequence_splitting,
            batches_per_file=self.config.batches_per_file,
        )
        if not self.config.reprocess and manifest_path.is_file():
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
            previous_partitioning = previous_manifest.pop("window_partitioning", None)
            if previous_partitioning not in (None, partitioning):
                raise ValueError(
                    "Cannot resume with different window partitioning settings"
                )
            if _stable_json_value(previous_manifest) != _stable_json_value(manifest):
                raise ValueError(
                    "Cannot continue preprocessing with a different preprocessing "
                    "manifest. Check sequence layout, input path, selected/data "
                    "columns, output format, mask/metadata settings, split/stride "
                    "settings, max_rows, process_by_file, prediction alignment, "
                    "or metadata/maps/statistics."
                )
        elif not self.config.reprocess and (
            (scratch / "depth-shards.json").exists()
            or any(scratch.glob("*.complete.json"))
        ):
            raise ValueError(
                "Cannot continue preprocessing because the temp manifest is missing."
            )

        if not self.config.depth_layouts:
            manifest["window_partitioning"] = partitioning
        write_json_atomic(manifest_path, manifest, indent=4, allow_nan=True)

    @beartype
    def _restore_resume_manifest_state(
        self,
        selected_columns: Optional[list[str]],
        write_format: str,
    ) -> None:
        """Validate a completed resume and restore metadata-generation state."""
        manifest_path = os.path.join(
            self.config.project_root,
            "data",
            self.target_dir,
            "preprocess-manifest.json",
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

        self.output_features = FittedColumns.from_metadata(
            previous["data_columns"], previous
        )
        self._write_or_validate_resume_manifest(
            selected_columns, write_format, self.output_features
        )
        self.has_sample_positions = bool(
            curriculum_columns(previous.get("curriculum_column"))
        )

    @beartype
    def _export_metadata(self, fitted: FittedColumns) -> None:
        """Write metadata config JSON for training/inference."""
        if self.image_time_rows:
            pl.DataFrame(
                self.image_time_rows,
                schema=["sequenceId", "itemPosition", "timeMs"],
                orient="row",
            ).write_parquet(self.output.time_index_path())
        self.output_features = fitted
        data_driven_config = {
            **fitted.metadata(),
            "cardinality_config": self.cardinality_config | self.static_cardinality,
            "special_token_ids": SPECIAL_TOKEN_IDS.ids_by_label,
            **self.config.split_metadata(),
            "split_paths": [
                os.path.splitext(split_path)[0]
                if not self.config.merge_output
                else split_path
                for split_path in self.split_paths
            ],
            "normalize_real_columns": self.normalize_real_columns,
            "normalize_on_all_data": self.metadata_fitted_on_all_data,
            **self.image_manifest,
            "raw_mask_column": self.config.mask_column,
            "raw_curriculum_column": self.config.curriculum_column,
            **self._layout_metadata(),
        }
        self.output.write_metadata(data_driven_config, fitted.data_columns)

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
                        layouts=self.config.depth_layouts,
                        n_classes=self.output_features.n_classes,
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

        from sequifier.io.preprocessing_output import WindowOutput

        assert isinstance(self.output, WindowOutput)
        metadata = {
            "split_index": split_index,
            "source_observations_path": str(self.output.source_path),
            **self.output_features.metadata(),
            "normalize_real_columns": self.normalize_real_columns,
            **self.image_manifest,
            "total_samples": total_samples,
            "batch_files": batch_files_metadata,
            **self._layout_metadata(),
        }
        metadata["window_length"] = self.config.window_lengths[split_index]
        if self.has_sample_positions:
            metadata["curriculum_column"] = self.config.curriculum_column

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
            minimum, maximum = cast(int, values.min()), cast(int, values.max())
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


def _process_batches_multiple_files_inner(
    plan,
    features,
    selected_columns,
    normalize,
    cardinality,
    num_workers,
    process_id,
    file_paths,
):
    """Apply the fitted transform and write each file in this worker's shard."""
    config = plan.config
    outputs = []
    scratch = Path(config.project_root) / "data" / plan.target_dir
    if not file_paths:
        raise ValueError("No files found to process.")
    pad_width = len(str(len(file_paths) - 1))
    rows_read = 0
    with (
        _create_preprocess_pool(num_workers) if num_workers > 1 else nullcontext()
    ) as pool:
        for index, path in enumerate(file_paths):
            remaining = None if config.max_rows is None else config.max_rows - rows_read
            if remaining is not None and remaining <= 0:
                break
            file_index = str(index).zfill(pad_width)
            name = f"{plan.name}-{process_id}-{file_index}"
            file_plan = plan.named(name)
            checkpoint = scratch / f"{name}.complete.json"
            if not config.reprocess and checkpoint.is_file():
                saved = json.loads(checkpoint.read_text())
                if all((scratch / part).is_file() for part in saved["parts"]):
                    rows_read += saved["rows"]
                    if config.merge_output:
                        outputs.append(file_plan.intermediate_paths)
                    logger.info(f"Skipping already processed file: {path}")
                    continue
            data = _load_and_preprocess_data(
                path,
                config.read_format,
                selected_columns,
                remaining,
                config.mask_column,
                config.curriculum_column,
                split_column=config.split_column,
                split_values=config.split_values,
            )
            rows_read += data.height
            from sequifier.io.preprocessing_output import store_source_rows

            store_source_rows(
                scratch,
                name,
                with_domain_record(
                    data, features.data_columns, DOMAIN_OBSERVATIONS, config.mask_column
                ),
            )
            data = features.transform(
                data,
                normalize,
                cardinality,
                config.mask_column,
                input_types=features.col_types,
            )
            shards = _process_batches_single_file(file_plan, data, num_workers, pool)
            if config.merge_output:
                paths = file_plan.intermediate_paths
                merge_window_outputs(shards, paths, config.write_format)
                outputs.append(paths)
            else:
                paths = [path for batch_paths in shards for path in batch_paths]
            write_json_atomic(
                checkpoint,
                {
                    "rows": data.height,
                    "parts": [Path(part).name for part in paths],
                },
            )
    return outputs


@beartype
def _process_batches_single_file(
    plan: WindowWritePlan,
    data: pl.DataFrame,
    num_workers: Optional[int],
    worker_pool: Optional[Any] = None,
) -> list[list[str]]:
    """Split one file into worker batches and preprocess them."""
    config = plan.config
    if config.split_method == "value_cutoff" and data.height > 1:
        sequence_ids = data.get_column("sequenceId").to_numpy()
        values = data.get_column(SPLIT_VALUE_COLUMN).to_numpy()
        if np.any((sequence_ids[1:] == sequence_ids[:-1]) & (values[1:] < values[:-1])):
            raise ValueError(
                "split_column values must be non-decreasing within each sequence"
            )
    num_workers = num_workers or multiprocessing.cpu_count()
    sequence_count = len(_sequence_run_bounds(data)[0])
    maximum_batches = data.height if config.allow_sequence_splitting else sequence_count
    num_workers = min(num_workers, maximum_batches)
    if (
        not config.normalize_on_all_data
        and config.allow_sequence_splitting
        and config.split_method == "within_sequence"
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
    batch_limits = get_batch_limits(data, num_workers, config.allow_sequence_splitting)
    valid_batch_limits = [(s, e) for s, e in batch_limits if (e - s) > 0]
    batches = [
        (plan, process_id, data.slice(start, end - start))
        for process_id, (start, end) in enumerate(valid_batch_limits)
    ]

    if len(batches) == 1:
        return [preprocess_batch(*batches[0])]
    if worker_pool is not None:
        return worker_pool.starmap(preprocess_batch, batches)
    with _create_preprocess_pool(len(batches)) as pool:
        return pool.starmap(preprocess_batch, batches)


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
) -> str:
    """Write one accumulated PT shard without constructing long-format rows."""
    split_path_batch_seq = split_path.replace(
        ".pt", f"-{process_id}-{file_index_str}.pt"
    )
    out_path = insert_top_folder(split_path_batch_seq, target_dir)
    process_and_write_sequence_windows_pt(windows_to_write, out_path, col_types)
    _write_window_shard_summary(out_path, windows_to_write)
    return out_path


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
) -> str:
    """Write dense windows through the compatibility long-format Parquet adapter."""
    combined_df = _sequence_windows_to_long_dataframe(windows_to_write, schema)
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
    return out_path


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
                str(path.with_stem(f"{path.stem}-{process_id}")), target_dir
            )
            for path in map(Path, split_paths)
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
        data = _sequence_windows_to_long_dataframe(windows, self.schemas[group])
        if self.write_format == "parquet":
            self._writers[group].write_table(data.to_arrow())
        else:
            data.write_csv(self._writers[group], include_header=False)

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()


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
    config: PreprocessorModel,
    plan: SplitWindowPlan,
    assignments: dict[int, int] | None,
) -> dict[int, tuple[WindowPlacement, np.ndarray | None]]:
    sequence_id = int(batch_arrays.sequence_ids[run_start])
    length = run_stop - run_start
    assigned = None
    bounds = None
    if config.split_method == "between_sequence":
        assigned = (assignments or {}).get(sequence_id)
        if assigned is None:
            assigned = assign_sequence_to_split(
                sequence_id, _require_split_ratios(config.split_ratios), config.seed
            )
    elif config.split_method == "value_cutoff":
        cutoffs = _normalized_split_cutoffs(config.split_values)
        if cutoffs is None or batch_arrays.split_values is None:
            raise ValueError("value_cutoff splitting requires split_values")
        values = batch_arrays.split_values[run_start:run_stop]
        bounds = cutoff_intervals(values, cutoffs)
    first = int(batch_arrays.item_positions[run_start])
    curriculum = {
        name: values[run_start:run_stop]
        for name, values in batch_arrays.curriculum_values.items()
    }
    return {
        split: (placement, placement.curriculum(curriculum, first, sequence_id))
        for split, placement in plan.placements(
            first, length, assigned=assigned, bounds=bounds
        )
    }


@beartype
def preprocess_batch(
    plan: WindowWritePlan, process_id: int, batch: pl.DataFrame
) -> list[str]:
    """Extract and write all split windows for one batch."""
    config = plan.config
    scratch = Path(config.project_root) / "data" / plan.target_dir
    checkpoint = scratch / f"{plan.name}-batch-{process_id}.complete.json"
    if not config.reprocess and checkpoint.is_file():
        parts = [str(scratch / part) for part in json.loads(checkpoint.read_text())]
        if all(Path(part).is_file() for part in parts):
            return parts
    batch_arrays = _batch_to_arrays(batch, plan.columns)
    split_schemas = [
        _schema_for_split(plan.schema, width) for width in plan.windows.widths
    ]
    sequence_count = len(batch_arrays.run_starts)

    def extract_for_run(
        run_start: int, run_stop: int
    ) -> dict[int, Optional[SequenceWindows]]:
        placements = _place_sequence_windows_for_splits(
            batch_arrays, run_start, run_stop, config, plan.windows, plan.assignments
        )
        return {
            split: _dense_windows(
                batch_arrays, run_start, run_stop, placement, samples, plan.columns
            )
            for split, (placement, samples) in placements.items()
        }

    writers = (
        _MergedLongOutputWriters(
            plan.split_paths,
            process_id,
            plan.target_dir,
            config.write_format,
            split_schemas,
        )
        if config.merge_output
        else None
    )
    file_indices = [0] * len(plan.split_paths)
    shard_paths = []
    pad_width = len(str(max(1, sequence_count)))
    windows_by_split: list[list[SequenceWindows]] = [[] for _ in plan.split_paths]
    buffered_bytes = [0] * len(plan.split_paths)

    def flush(group: int) -> None:
        pending = windows_by_split[group]
        if not pending:
            return
        if writers is not None:
            writers.append(group, pending)
        else:
            file_index = str(file_indices[group]).zfill(pad_width)
            if config.write_format == "pt":
                path = _write_accumulated_windows_pt(
                    pending,
                    plan.split_paths[group],
                    process_id,
                    file_index,
                    plan.target_dir,
                    plan.col_types,
                )
            else:
                path = _write_accumulated_windows_parquet(
                    pending,
                    plan.split_paths[group],
                    process_id,
                    file_index,
                    plan.target_dir,
                    split_schemas[group],
                )
            file_indices[group] += 1
            shard_paths.append(path)
        pending.clear()
        buffered_bytes[group] = 0

    try:
        for run_start, run_stop in zip(batch_arrays.run_starts, batch_arrays.run_stops):
            windows = extract_for_run(int(run_start), int(run_stop))
            for group, split_windows in windows.items():
                if split_windows is None or not split_windows.n_samples:
                    continue
                windows_by_split[group].append(split_windows)
                buffered_bytes[group] += _sequence_windows_nbytes(split_windows)
                if (
                    len(windows_by_split[group]) >= config.batches_per_file
                    or buffered_bytes[group] >= MAX_WINDOW_BUFFER_BYTES
                ):
                    flush(group)
        for group in range(len(plan.split_paths)):
            flush(group)
    finally:
        if writers is not None:
            writers.close()
    parts = writers.paths if writers is not None else shard_paths
    write_json_atomic(checkpoint, [Path(part).name for part in parts])
    return parts


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
    windows = extract_sequence_windows(data, layout, stride_for_split, columns)
    return _sequence_windows_to_long_dataframe(
        [windows] if windows is not None and windows.n_samples else [], schema
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


def merge_window_outputs(
    shards: list[list[str]], destinations: list[str], write_format: str
) -> None:
    """Merge each split in worker order, then release the intermediate files."""
    for destination, paths in zip(destinations, zip(*shards, strict=True), strict=True):
        logger.info(f"writing to: {destination}")
        if write_format == "csv":
            with open(destination, "wb") as stream:
                subprocess.run(["csvstack", *paths], stdout=stream, check=True)
        else:
            combine_parquet_files(list(paths), destination)
    for paths in shards:
        for path in paths:
            os.remove(path)


@beartype
def combine_parquet_files(files: list[str], out_path: str) -> None:
    """Stream-concatenate Parquet files with the first file schema."""
    schema = pq.ParquetFile(files[0]).schema_arrow
    with pq.ParquetWriter(out_path, schema=schema, compression="snappy") as writer:
        for file in files:
            parquet_file = pq.ParquetFile(file)
            for batch in parquet_file.iter_batches(batch_size=262_144):
                writer.write_batch(batch)
