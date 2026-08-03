import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from beartype import beartype
from loguru import logger

from sequifier.config.preprocess_config import load_preprocessor_config
from sequifier.helpers import (
    PANDAS_TO_TORCH_TYPES,
    StoredWindowLayout,
    assign_sequence_to_split,
    canonicalize_polars_dtype_name,
    is_float_dtype_name,
    is_integer_dtype_name,
    polars_dtype_from_name,
    read_data,
    write_data,
)
from sequifier.special_tokens import (
    SPECIAL_TOKEN_ID_VALUES,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKEN_LABELS,
    validate_special_token_ids,
)

INPUT_METADATA_COLUMNS = ("sequenceId", "itemPosition")
REAL_MASK_VALUE = 0.0
CURRENT_STORED_WINDOW_LAYOUT_VERSION = 2

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


def _stable_json_digest(value: Any) -> str:
    encoded = json.dumps(
        _stable_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@beartype
def _normalize_column_types(
    column_types: Optional[dict[str, str]],
) -> Optional[dict[str, str]]:
    if column_types is None:
        return None
    return {
        column: canonicalize_polars_dtype_name(dtype)
        for column, dtype in column_types.items()
    }


@beartype
def _configured_column_types_for_data_columns(
    column_types: Optional[dict[str, str]],
    data_columns: list[str],
) -> Optional[dict[str, str]]:
    if column_types is None:
        return None

    missing_columns = [column for column in data_columns if column not in column_types]
    if missing_columns:
        raise ValueError(
            "column_types must include every to-be-processed column. "
            f"Missing: {missing_columns}"
        )

    return {column: column_types[column] for column in data_columns}


@beartype
def _dtype_is_numeric(dtype: Any) -> bool:
    return dtype.is_numeric() if hasattr(dtype, "is_numeric") else False


@beartype
def _apply_configured_input_casting(
    data: pl.DataFrame,
    data_columns: list[str],
    column_types: Optional[dict[str, str]],
) -> pl.DataFrame:
    """Cast input columns early when the requested type defines processing semantics."""
    configured = _configured_column_types_for_data_columns(column_types, data_columns)
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
        pl.col(column).cast(polars_dtype_from_name(col_types[column]))
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
def _resolve_unified_parquet_type(column_types: dict[str, str]) -> Any:
    if not column_types:
        raise ValueError("column_types cannot be empty")

    normalized_types = [
        canonicalize_polars_dtype_name(type_) for type_ in column_types.values()
    ]
    float_types = [type_ for type_ in normalized_types if is_float_dtype_name(type_)]
    integer_types = [
        type_ for type_ in normalized_types if is_integer_dtype_name(type_)
    ]

    if not float_types:
        if len(set(integer_types)) > 1:
            logger.warning(
                "Multiple integer column_types were specified for Parquet output; "
                "using a unified integer schema."
            )
        return _resolve_integer_sequence_type(integer_types)

    resolved_float = _highest_ranked_type(float_types, FLOAT_TYPE_ORDER)
    if len(set(normalized_types)) > 1:
        logger.warning(
            "Multiple column_types were specified for Parquet output; "
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
def _resolve_pt_extraction_type(column_types: dict[str, str]) -> Any:
    normalized_types = [
        canonicalize_polars_dtype_name(type_) for type_ in column_types.values()
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


class Preprocessor:
    """Stateful preprocessing pipeline for single-file or folder inputs."""

    @beartype
    def __init__(
        self,
        project_root: str,
        continue_preprocessing: bool,
        data_path: str,
        read_format: str,
        write_format: str,
        merge_output: bool,
        allow_sequence_splitting: bool,
        selected_columns: Optional[list[str]],
        split_ratios: list[float],
        stored_context_width: int,
        stride_by_split: list[int],
        max_rows: Optional[int],
        seed: int,
        n_cores: Optional[int],
        batches_per_file: int,
        process_by_file: bool,
        subsequence_start_mode: str,
        use_precomputed_maps: Optional[list[str]],
        metadata_config_path: Optional[str],
        max_target_offset: int = 1,
        mask_column: Optional[str] = None,
        column_types: Optional[dict[str, str]] = None,
        split_method: str = "within_sequence",
        normalize_real_columns: bool = True,
    ):
        """Initialize and run preprocessing from validated config fields."""
        self.project_root = project_root
        self.batches_per_file = batches_per_file
        self.data_path = data_path
        self.read_format = read_format
        self.write_format = write_format

        self.data_name_root = os.path.splitext(os.path.basename(data_path))[0]
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

        self.use_precomputed_maps = use_precomputed_maps
        self.metadata_config_path = metadata_config_path
        self.mask_column = mask_column
        if split_method not in ["within_sequence", "between_sequence"]:
            raise ValueError(
                "split_method must be one of 'within_sequence', 'between_sequence'"
            )
        self.split_method = split_method
        self.split_ratios = split_ratios
        self.stride_by_split = stride_by_split
        self.max_rows = max_rows
        self.process_by_file = process_by_file
        self.subsequence_start_mode = subsequence_start_mode
        self.column_types = _normalize_column_types(column_types)
        self.normalize_real_columns = normalize_real_columns
        if self.mask_column is not None and self.metadata_config_path is None:
            raise ValueError("metadata_config_path must be set when mask_column is set")

        self.seed = seed
        np.random.seed(seed)
        self.n_cores = n_cores or multiprocessing.cpu_count()
        self.continue_preprocessing = continue_preprocessing
        self.storage_layout = StoredWindowLayout(
            stored_context_width=stored_context_width,
            max_target_offset=max_target_offset,
            version=CURRENT_STORED_WINDOW_LAYOUT_VERSION,
        )
        self._setup_directories()

        if selected_columns is not None:
            selected_columns = ["sequenceId", "itemPosition"] + selected_columns
            if self.mask_column is not None and self.mask_column in selected_columns:
                raise ValueError(
                    f"'{self.mask_column}' is not allowed to be in 'selected_columns'"
                )

        self._setup_split_paths(write_format, len(split_ratios))

        if self.continue_preprocessing:
            if self.merge_output:
                paths_to_check = self.split_paths
            else:
                paths_to_check = [
                    os.path.join(
                        self.project_root, "data", f"{self.data_name_root}-split{i}"
                    )
                    for i in range(len(split_ratios))
                ]

            if any(os.path.exists(p) for p in paths_to_check):
                logger.info(
                    "Existing split paths found with continue_preprocessing=True. "
                    "Skipping processing and running cleanup."
                )
                self._cleanup(write_format)
                return

        if os.path.isfile(data_path):
            data = _load_and_preprocess_data(
                data_path,
                read_format,
                selected_columns,
                max_rows,
                self.mask_column,
            )
            data_columns = _get_data_columns(data, self.mask_column)
            configured_col_types = _configured_column_types_for_data_columns(
                self.column_types, data_columns
            )
            data = _apply_configured_input_casting(
                data, data_columns, configured_col_types
            )
            if self.metadata_config_path:
                metadata_path = os.path.join(
                    self.project_root, self.metadata_config_path
                )

                with open(metadata_path, "r") as f:
                    preexisting_metadata = json.load(f)

                validate_special_token_ids(
                    preexisting_metadata["special_token_ids"],
                    source=f"metadata config '{self.metadata_config_path}'",
                )
                id_maps = preexisting_metadata["id_maps"]
                selected_columns_statistics = preexisting_metadata[
                    "selected_columns_statistics"
                ]
                n_classes = preexisting_metadata["n_classes"]
                col_types = preexisting_metadata["column_types"]
            else:
                id_maps, selected_columns_statistics = {}, {}

                precomputed_id_maps = load_precomputed_id_maps(
                    self.project_root, data_columns, self.use_precomputed_maps
                )

                id_maps, selected_columns_statistics = _get_column_statistics(
                    data,
                    data_columns,
                    id_maps,
                    selected_columns_statistics,
                    0,
                    precomputed_id_maps,
                    self.mask_column,
                )

                id_maps = id_maps | precomputed_id_maps
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

            schema = self._create_schema(
                col_types, self.storage_layout.stored_context_width
            )

            data = data.sort(["sequenceId", "itemPosition"])
            n_batches = _process_batches_single_file(
                self.project_root,
                self.data_name_root,
                data,
                schema,
                self.n_cores,
                self.storage_layout,
                stride_by_split,
                data_columns,
                col_types,
                split_ratios,
                write_format,
                self.split_paths,
                self.target_dir,
                self.batches_per_file,
                subsequence_start_mode,
                self.merge_output,
                self.allow_sequence_splitting,
                self.split_method,
                self.seed,
            )

            if self.merge_output:
                input_files = create_file_paths_for_single_file(
                    self.project_root,
                    self.target_dir,
                    len(split_ratios),
                    n_batches,
                    self.data_name_root,
                    write_format,
                )
                combine_multiprocessing_outputs(
                    self.project_root,
                    self.target_dir,
                    len(split_ratios),
                    input_files,
                    self.data_name_root,
                    write_format,
                    in_target_dir=False,
                )
                delete_files(input_files)
        else:
            if self.metadata_config_path:
                metadata_path = os.path.join(
                    self.project_root, self.metadata_config_path
                )

                with open(metadata_path, "r") as f:
                    preexisting_metadata = json.load(f)

                validate_special_token_ids(
                    preexisting_metadata["special_token_ids"],
                    source=f"metadata config '{self.metadata_config_path}'",
                )
                id_maps = preexisting_metadata["id_maps"]
                selected_columns_statistics = preexisting_metadata[
                    "selected_columns_statistics"
                ]
                n_classes = preexisting_metadata["n_classes"]
                col_types = preexisting_metadata["column_types"]

                # Reconstruct data_columns from the provided col_types
                data_columns = [
                    col
                    for col in col_types.keys()
                    if col not in _reserved_input_columns(self.mask_column)
                ]
                configured_col_types = _configured_column_types_for_data_columns(
                    self.column_types, data_columns
                )
                if configured_col_types is not None:
                    col_types = configured_col_types

                # We still need to find the files to process
                files_to_process = []
                for root, dirs, files in os.walk(data_path):
                    for file in sorted(list(files)):
                        if file.endswith(read_format):
                            files_to_process.append(os.path.join(root, file))
            else:
                (
                    files_to_process,
                    n_classes,
                    id_maps,
                    selected_columns_statistics,
                    col_types,
                    data_columns,
                ) = self._get_column_metadata_across_files(
                    data_path,
                    read_format,
                    max_rows,
                    selected_columns,
                    self.column_types,
                )
                for col in id_maps:
                    if self.column_types is None:
                        col_types[col] = "Int64"

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
            schema = self._create_schema(
                col_types, self.storage_layout.stored_context_width
            )

            self._process_batches_multiple_files(
                files_to_process,
                read_format,
                selected_columns,
                max_rows,
                schema,
                self.n_cores,
                self.storage_layout,
                stride_by_split,
                data_columns,
                n_classes,
                id_maps,
                selected_columns_statistics,
                col_types,
                split_ratios,
                write_format,
                process_by_file,
                subsequence_start_mode,
            )

        self._cleanup(write_format)

    @beartype
    def _create_schema(
        self, col_types: dict[str, str], stored_context_width: int
    ) -> dict[str, Any]:
        """Build the long-format extracted-window schema."""
        schema = {
            "sequenceId": pl.Int64,
            "subsequenceId": pl.Int64,
            "startItemPosition": pl.Int64,
            "leftPadLength": pl.Int64,
            "inputCol": pl.String,
        }

        if self.write_format == "parquet":
            sequence_position_type = _resolve_unified_parquet_type(col_types)
        else:
            sequence_position_type = _resolve_pt_extraction_type(col_types)

        schema.update(
            {
                str(i): sequence_position_type
                for i in range(stored_context_width - 1, -1, -1)
            }
        )

        return schema

    @beartype
    def _get_column_metadata_across_files(
        self,
        data_path: str,
        read_format: str,
        max_rows: Optional[int],
        selected_columns: Optional[list[str]],
        column_types: Optional[dict[str, str]],
    ) -> tuple[
        list[str],
        dict[str, int],
        dict[str, dict[Union[str, int], int]],
        dict[str, dict[str, float]],
        dict[str, str],
        list[str],
    ]:
        """Accumulate metadata/statistics over a folder input."""

        n_rows_running_count = 0
        id_maps, selected_columns_statistics = {}, {}

        col_types, data_columns = None, None

        precomputed_id_maps = load_precomputed_id_maps(
            self.project_root, data_columns, self.use_precomputed_maps
        )

        files_to_process = []
        logger.info(f"Data path: {data_path}")
        for root, dirs, files in os.walk(data_path):
            logger.info(f"N Files : {len(files)}")
            for file in sorted(list(files)):
                if file.endswith(read_format) and (
                    max_rows is None or n_rows_running_count < max_rows
                ):
                    logger.info(f"Preprocessing: reading {file}")
                    files_to_process.append(os.path.join(root, file))
                    max_rows_inner = (
                        None
                        if max_rows is None
                        else max(0, max_rows - n_rows_running_count)
                    )
                    data = _load_and_preprocess_data(
                        os.path.join(root, file),
                        read_format,
                        selected_columns,
                        max_rows_inner,
                        self.mask_column,
                    )

                    current_file_cols = _get_data_columns(data, self.mask_column)
                    current_configured_col_types = (
                        _configured_column_types_for_data_columns(
                            column_types, current_file_cols
                        )
                    )
                    data = _apply_configured_input_casting(
                        data, current_file_cols, current_configured_col_types
                    )

                    if col_types is None:
                        data_columns = current_file_cols
                        col_types = current_configured_col_types or {
                            col: str(data.schema[col]) for col in data_columns
                        }

                        for col in precomputed_id_maps.keys():
                            if col not in data_columns:
                                raise ValueError(
                                    f"Precomputed column {col} not found in {file}"
                                )
                    else:
                        if set(current_file_cols) != set(col_types.keys()):
                            missing = set(col_types.keys()) - set(current_file_cols)
                            extra = set(current_file_cols) - set(col_types.keys())
                            raise ValueError(
                                f"Schema mismatch in file '{file}'.\n"
                                f"Expected columns: {list(col_types.keys())}\n"
                                f"Found columns: {current_file_cols}\n"
                                f"Missing: {missing}\n"
                                f"Extra: {extra}"
                            )

                        if column_types is None:
                            for col in current_file_cols:
                                if str(data.schema[col]) != col_types[col]:
                                    raise ValueError(
                                        f"Type mismatch for column '{col}' in file '{file}'. "
                                        f"Expected {col_types[col]}, got {str(data.schema[col])}"
                                    )

                    if data_columns is None:
                        raise ValueError("data_columns is None")

                    id_maps, selected_columns_statistics = _get_column_statistics(
                        data,
                        data_columns,
                        id_maps,
                        selected_columns_statistics,
                        n_rows_running_count,
                        precomputed_id_maps,
                        self.mask_column,
                    )
                    n_rows_running_count += data.shape[0]

        id_maps = id_maps | precomputed_id_maps

        if data_columns is None:
            raise RuntimeError("data_columns was not initialized correctly.")
        n_classes = {col: max(id_maps[col].values()) + 1 for col in id_maps}

        if col_types is None:
            raise RuntimeError("col_types was not initialized correctly.")
        files_to_process = sorted(files_to_process)

        return (
            files_to_process,
            n_classes,
            id_maps,
            selected_columns_statistics,
            col_types,
            data_columns,
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
    def _process_batches_multiple_files(
        self,
        file_paths: list[str],
        read_format: str,
        selected_columns: Optional[list[str]],
        max_rows: Optional[int],
        schema: Any,
        n_cores: int,
        layout: StoredWindowLayout,
        stride_by_split: list[int],
        data_columns: list[str],
        n_classes: dict[str, int],
        id_maps: dict[str, dict[Union[int, str], int]],
        selected_columns_statistics: dict[str, dict[str, float]],
        col_types: dict[str, str],
        split_ratios: list[float],
        write_format: str,
        process_by_file: bool = True,
        subsequence_start_mode: str = "distribute",
        mask_column: Optional[str] = None,
    ) -> None:
        """Dispatch folder preprocessing by file or by process shard."""
        if mask_column is None:
            mask_column = self.mask_column

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
                stride_by_split=stride_by_split,
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
                subsequence_start_mode=subsequence_start_mode,
                mask_column=mask_column,
                split_method=self.split_method,
                seed=self.seed,
                normalize_real_columns=self.normalize_real_columns,
            )
            input_files = create_file_paths_for_multiple_files2(
                self.project_root,
                self.target_dir,
                len(split_ratios),
                1,
                {0: len(file_paths)},
                self.data_name_root,
                write_format,
            )
        else:
            assert process_by_file is False
            n_file_sets = (len(file_paths) // n_cores) + 1

            file_sets = [
                file_paths[i : i + n_file_sets]
                for i in range(0, len(file_paths), n_file_sets)
            ]

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
                "stride_by_split": stride_by_split,
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
                "subsequence_start_mode": subsequence_start_mode,
                "mask_column": mask_column,
                "split_method": self.split_method,
                "seed": self.seed,
                "normalize_real_columns": self.normalize_real_columns,
            }

            job_params = [
                list(kwargs_1.values())
                + [process_id, file_set]
                + list(kwargs_2.values())
                for process_id, file_set in enumerate(file_sets)
            ]
            logger.info(f"_process_batches_multiple_files n_cores: {n_cores}")
            logger.info(f"_process_batches_multiple_files {len(job_params) = }")

            with multiprocessing.get_context("spawn").Pool(
                processes=len(job_params)
            ) as pool:
                pool.starmap(_process_batches_multiple_files_inner, job_params)

            input_files = create_file_paths_for_multiple_files2(
                self.project_root,
                self.target_dir,
                len(split_ratios),
                len(job_params),
                {i: len(file_sets[i]) for i in range(len(file_sets))},
                self.data_name_root,
                write_format,
            )
        if self.merge_output:
            combine_multiprocessing_outputs(
                self.project_root,
                self.target_dir,
                len(split_ratios),
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

                pattern = re.compile(rf".+split{i}-\d+-\d+\.\w+")

                for file_path in directory.iterdir():
                    if file_path.is_file() and pattern.match(file_path.name):
                        destination = Path(folder_path) / file_path.name
                        logger.info(f"Moving '{file_path}' to '{destination}'")
                        shutil.move(str(file_path), str(destination))

                self._create_metadata_for_folder(folder_path, write_format)

        if not os.listdir(directory) or self.target_dir == "temp":
            shutil.rmtree(directory)

    @beartype
    def _layout_metadata(self) -> dict[str, int]:
        return {
            "stored_context_width": self.storage_layout.stored_context_width,
            "max_target_offset": self.storage_layout.max_target_offset,
            "stored_window_layout_version": self.storage_layout.version,
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
                "split_ratios": self.split_ratios,
                "split_method": self.split_method,
                "seed": self.seed,
                "stride_by_split": self.stride_by_split,
                "max_rows": self.max_rows,
                "process_by_file": self.process_by_file,
                "subsequence_start_mode": self.subsequence_start_mode,
                "mask_column": self.mask_column,
                "use_precomputed_maps": self.use_precomputed_maps,
                "n_classes": n_classes,
                "id_maps": id_maps,
                "column_types": col_types,
                "selected_columns_statistics": selected_columns_statistics,
                "normalize_real_columns": self.normalize_real_columns,
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
            if _stable_json_value(previous_manifest) != _stable_json_value(manifest):
                raise ValueError(
                    "Cannot continue preprocessing with a different preprocessing "
                    "manifest. Check sequence layout, input path, selected/data "
                    "columns, output format, mask/metadata settings, split/stride "
                    "settings, max_rows, process_by_file, subsequence_start_mode, "
                    "or metadata/maps/statistics."
                )
            return

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)

    @beartype
    def _export_metadata(
        self,
        id_maps: dict[str, dict[Union[str, int], int]],
        n_classes: dict[str, int],
        col_types: dict[str, str],
        selected_columns_statistics: dict[str, dict[str, float]],
    ) -> None:
        """Write metadata config JSON for training/inference."""
        data_driven_config = {
            "n_classes": n_classes,
            "id_maps": id_maps,
            "special_token_ids": SPECIAL_TOKEN_IDS.ids_by_label,
            "split_paths": [
                os.path.splitext(split_path)[0] if not self.merge_output else split_path
                for split_path in self.split_paths
            ],
            "column_types": col_types,
            "selected_columns_statistics": {
                col: {"mean": stats["mean"], "std": stats["std"]}
                for col, stats in selected_columns_statistics.items()
            },
            "normalize_real_columns": self.normalize_real_columns,
            "stride_by_split": self.stride_by_split,
            "subsequence_start_mode": self.subsequence_start_mode,
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
    def _create_metadata_for_folder(self, folder_path: str, write_format: str) -> None:
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
                if write_format == "pt":
                    sequences_dict, _, _, _, left_pad_lengths = torch.load(
                        file_path, weights_only=False
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
                        left_pad_lengths = (
                            lazy_df.group_by(["sequenceId", "subsequenceId"])
                            .agg(pl.col("leftPadLength").first())
                            .select("leftPadLength")
                            .collect()
                            .get_column("leftPadLength")
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
                            }
                        )
                        total_samples += n_samples
            except Exception as e:
                logger.warning(f"Could not process file {file_path} for metadata: {e}")

        metadata = {
            "total_samples": total_samples,
            "batch_files": batch_files_metadata,
            **self._layout_metadata(),
        }

        metadata_path = directory / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)


@beartype
def _reserved_input_columns(mask_column: Optional[str]) -> tuple[str, ...]:
    if mask_column is None:
        return INPUT_METADATA_COLUMNS
    return (*INPUT_METADATA_COLUMNS, mask_column)


@beartype
def _get_data_columns(
    data: pl.DataFrame, mask_column: Optional[str] = None
) -> list[str]:
    return [
        col for col in data.columns if col not in _reserved_input_columns(mask_column)
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
) -> Optional[list[str]]:
    if selected_columns is None or mask_column is None:
        return selected_columns

    if read_format != "parquet":
        return _deduplicate_columns(selected_columns + [mask_column])

    schema_columns = pq.read_schema(data_path).names
    if mask_column in schema_columns:
        return _deduplicate_columns(selected_columns + [mask_column])
    raise ValueError(f"mask_column '{mask_column}' not found in {data_path}")


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
) -> tuple[pl.DataFrame, dict[str, int], dict[str, str]]:
    """Apply categorical maps and optional numeric standardization."""
    col_types_was_provided = col_types is not None

    if n_classes is None:
        n_classes = {col: max(id_maps[col].values()) + 1 for col in id_maps}

    if col_types is None:
        col_types = {col: str(data.schema[col]) for col in data_columns}

    missing_columns = [
        col
        for col in data_columns
        if col not in id_maps and col not in selected_columns_statistics
    ]
    if missing_columns:
        raise ValueError(
            "No unmasked examples found for columns: "
            f"{missing_columns}. Check the mask column or provide precomputed metadata."
        )

    for col in data_columns:
        if col in id_maps:
            data = data.with_columns(pl.col(col).replace(id_maps[col], default=1))
            if not col_types_was_provided:
                col_types[col] = "Int64"
        elif col in selected_columns_statistics and normalize_real_columns:
            data = data.with_columns(
                (
                    (pl.col(col) - selected_columns_statistics[col]["mean"])
                    / (selected_columns_statistics[col]["std"] + 1e-9)
                ).alias(col)
            )

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

    for data_col in data_columns:
        dtype = data.schema[data_col]
        if isinstance(
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
        ):
            if data_col not in precomputed_id_maps:
                new_id_map = create_id_map(data, column=data_col)
                id_maps[data_col] = combine_maps(new_id_map, id_maps.get(data_col, {}))
            else:
                logger.info(f"Applying precomputed map for {data_col}")
        elif isinstance(dtype, (pl.Float16, pl.Float32, pl.Float64)):
            if data_col in precomputed_id_maps:
                raise ValueError(
                    f"Column {data_col} is not categorical, precomputed map is invalid."
                )

            chunk_mean = data.get_column(data_col).mean()
            chunk_std = data.get_column(data_col).std() or 0.0
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
        else:
            raise ValueError(f"Column {data_col} has unsupported dtype: {dtype}")

    return id_maps, selected_columns_statistics


@beartype
def _load_and_preprocess_data(
    data_path: str,
    read_format: str,
    selected_columns: Optional[list[str]],
    max_rows: Optional[int],
    mask_column: Optional[str] = None,
) -> pl.DataFrame:
    """Read, validate, column-filter, and row-limit one input file."""
    logger.info(f"Reading data from '{data_path}'...")
    columns_to_read = _selected_columns_with_optional_mask(
        data_path, read_format, selected_columns, mask_column
    )
    data = read_data(data_path, read_format, columns=columns_to_read)

    if mask_column is not None and mask_column not in data.columns:
        raise ValueError(f"mask_column '{mask_column}' not found in {data_path}")

    if data.null_count().sum().sum_horizontal().item() != 0:
        raise ValueError(f"NaN or null values not accepted: {data.null_count()}")

    if selected_columns:
        selected_columns_filtered = [
            col for col in selected_columns if col not in INPUT_METADATA_COLUMNS
        ]
        columns_to_select = list(INPUT_METADATA_COLUMNS) + selected_columns_filtered
        if mask_column is not None and mask_column in data.columns:
            columns_to_select.append(mask_column)
        data = data.select(_deduplicate_columns(columns_to_select))

    if max_rows:
        data = data.slice(0, int(max_rows))

    return data


def _check_file_has_been_processed(
    project_root: str,
    data_name_root: str,
    process_id: int,
    split_ratios: list[float],
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
        for split_index in range(len(split_ratios)):
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
    stride_by_split: list[int],
    data_columns: list[str],
    n_classes: dict[str, int],
    id_maps: dict[str, dict[Union[int, str], int]],
    selected_columns_statistics: dict[str, dict[str, float]],
    col_types: dict[str, str],
    split_ratios: list[float],
    write_format: str,
    split_paths: list[str],
    target_dir: str,
    batches_per_file: int,
    merge_output: bool,
    allow_sequence_splitting: bool,
    continue_preprocessing: bool,
    subsequence_start_mode: str,
    mask_column: Optional[str],
    split_method: str,
    seed: int,
    normalize_real_columns: bool,
):
    """Process this worker's file shard."""

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
                        split_ratios,
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
                        )
                        n_rows_running_count += data.shape[0]
                    continue

            data = _load_and_preprocess_data(
                path,
                read_format,
                selected_columns,
                max_rows_inner,
                mask_column,
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
                stride_by_split,
                data_columns,
                col_types,
                split_ratios,
                write_format,
                adjusted_split_paths,
                target_dir,
                batches_per_file,
                subsequence_start_mode,
                merge_output,
                allow_sequence_splitting,
                split_method,
                seed,
            )

            if merge_output:
                input_files = create_file_paths_for_multiple_files1(
                    project_root,
                    target_dir,
                    len(split_ratios),
                    n_batches,
                    process_id,
                    file_index_str,
                    data_name_root,
                    write_format,
                )
                combine_multiprocessing_outputs(
                    project_root,
                    target_dir,
                    len(split_ratios),
                    input_files,
                    data_name_root,
                    write_format,
                    in_target_dir=True,
                    pre_split_str=f"{process_id}-{file_index_str}",
                )

                delete_files(input_files)

            n_rows_running_count += data.shape[0]


@beartype
def _process_batches_single_file(
    project_root: str,
    data_name_root: str,
    data: pl.DataFrame,
    schema: Any,
    n_cores: Optional[int],
    layout: StoredWindowLayout,
    stride_by_split: list[int],
    data_columns: list[str],
    col_types: dict[str, str],
    split_ratios: list[float],
    write_format: str,
    split_paths: list[str],
    target_dir: str,
    batches_per_file: int,
    subsequence_start_mode: str,
    merge_output: bool,
    allow_sequence_splitting: bool,
    split_method: str = "within_sequence",
    seed: int = 1010,
) -> int:
    """Split one file into worker batches and preprocess them."""
    n_cores = n_cores or multiprocessing.cpu_count()
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
            stride_by_split,
            data_columns,
            col_types,
            split_ratios,
            target_dir,
            write_format,
            batches_per_file,
            subsequence_start_mode,
            merge_output,
            split_method,
            seed,
        )
        for process_id, (start, end) in enumerate(valid_batch_limits)
    ]

    if len(batches) > 1:
        with multiprocessing.get_context("spawn").Pool(processes=len(batches)) as pool:
            pool.starmap(preprocess_batch, batches)
    else:
        preprocess_batch(*batches[0])

    return len(batches)


@beartype
def get_combined_statistics(
    n1: int, mean1: float, std1: float, n2: int, mean2: float, std2: float
) -> tuple[float, float]:
    """Combine two mean/std summaries."""
    if n1 == 0:
        return mean2, std2
    if n2 == 0:
        return mean1, std1

    combined_mean = (n1 * mean1 + n2 * mean2) / (n1 + n2)

    if n1 + n2 <= 1:
        return combined_mean, 0.0

    sum_of_squares1 = (n1 - 1) * std1**2 + n1 * (mean1 - combined_mean) ** 2
    sum_of_squares2 = (n2 - 1) * std2**2 + n2 * (mean2 - combined_mean) ** 2

    combined_std = math.sqrt((sum_of_squares1 + sum_of_squares2) / (n1 + n2 - 1))

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
def process_and_write_data_pt(
    data: pl.DataFrame,
    stored_context_width: int,
    path: str,
    column_types: dict[str, str],
):
    """Write long-format sequences as packed PT tensors."""
    if data.is_empty():
        return

    sequence_cols = [str(c) for c in range(stored_context_width - 1, -1, -1)]

    all_feature_cols = data.get_column("inputCol").unique().to_list()

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

    aggregated_data = (
        data.group_by(["sequenceId", "subsequenceId"])
        .agg(aggs)
        .sort(["sequenceId", "subsequenceId"])
    )

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
    sequences_dict = {}

    for col_name in all_feature_cols:
        torch_dtype = PANDAS_TO_TORCH_TYPES[column_types[col_name]]

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
    torch.save(data_to_save, path)


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
    split_path_batch_seq = split_path.replace(
        f".{write_format}", f"-{process_id}-{file_index_str}.{write_format}"
    )
    out_path = insert_top_folder(split_path_batch_seq, target_dir)

    if write_format == "pt":
        process_and_write_data_pt(
            combined_df, layout.stored_context_width, out_path, col_types
        )
    elif write_format == "parquet":
        combined_df.write_parquet(out_path)


@beartype
def _extract_sequences_for_splits(
    data_subset: pl.DataFrame,
    sequence_id: int,
    schema: Any,
    layout: StoredWindowLayout,
    stride_by_split: list[int],
    data_columns: list[str],
    split_ratios: list[float],
    subsequence_start_mode: str,
    split_method: str,
    seed: int,
) -> dict[int, pl.DataFrame]:
    """Return extracted windows for one sequence across configured splits."""
    if split_method == "within_sequence":
        group_bounds = get_group_bounds(data_subset, split_ratios)
        return {
            i: cast_columns_to_string(
                extract_sequences(
                    data_subset.slice(lb, ub - lb),
                    schema,
                    layout,
                    stride_by_split[i],
                    data_columns,
                    subsequence_start_mode,
                )
            )
            for i, (lb, ub) in enumerate(group_bounds)
        }

    if split_method == "between_sequence":
        assigned_group = assign_sequence_to_split(sequence_id, split_ratios, seed)
        sequences = {i: pl.DataFrame(schema=schema) for i in range(len(split_ratios))}
        sequences[assigned_group] = cast_columns_to_string(
            extract_sequences(
                data_subset,
                schema,
                layout,
                stride_by_split[assigned_group],
                data_columns,
                subsequence_start_mode,
            )
        )
        return sequences

    raise ValueError(
        "split_method must be one of 'within_sequence', 'between_sequence'"
    )


@beartype
def preprocess_batch(
    project_root: str,
    data_name_root: str,
    process_id: int,
    batch: pl.DataFrame,
    schema: Any,
    split_paths: list[str],
    layout: StoredWindowLayout,
    stride_by_split: list[int],
    data_columns: list[str],
    col_types: dict[str, str],
    split_ratios: list[float],
    target_dir: str,
    write_format: str,
    batches_per_file: int,
    subsequence_start_mode: str,
    merge_output: bool,
    split_method: str = "within_sequence",
    seed: int = 1010,
) -> None:
    """Extract and write all split windows for one batch."""
    sequence_ids = sorted(batch.get_column("sequenceId").unique().to_list())

    if not merge_output:
        sequences_by_split = {i: [] for i in range(len(split_paths))}
        file_indices = {i: 0 for i in range(len(split_paths))}

        pad_width = len(str(math.ceil(len(sequence_ids) / batches_per_file) + 1))
        for i, sequence_id in enumerate(sequence_ids):
            data_subset = batch.filter(pl.col("sequenceId") == sequence_id)
            sequences = _extract_sequences_for_splits(
                data_subset,
                sequence_id,
                schema,
                layout,
                stride_by_split,
                data_columns,
                split_ratios,
                subsequence_start_mode,
                split_method,
                seed,
            )

            for group, split_df in sequences.items():
                if not split_df.is_empty():
                    sequences_by_split[group].append(split_df)

                # Check if the accumulator for this split has reached the desired size
                if len(sequences_by_split[group]) >= batches_per_file:
                    _write_accumulated_sequences(
                        sequences_by_split[group],
                        split_paths[group],
                        write_format,
                        process_id,
                        str(file_indices[group]).zfill(pad_width),
                        target_dir,
                        layout,
                        col_types,
                    )
                    # Reset the accumulator and increment the file index
                    sequences_by_split[group] = []
                    file_indices[group] += 1

        # After the loop, write any remaining sequences that didn't fill a full batch
        for group in range(len(split_paths)):
            _write_accumulated_sequences(
                sequences_by_split[group],
                split_paths[group],
                write_format,
                process_id,
                str(file_indices[group]).zfill(pad_width),
                target_dir,
                layout,
                col_types,
            )

    else:
        written_files: dict[int, list[str]] = {i: [] for i in range(len(split_paths))}
        for i, sequence_id in enumerate(sequence_ids):
            data_subset = batch.filter(pl.col("sequenceId") == sequence_id)
            sequences = _extract_sequences_for_splits(
                data_subset,
                sequence_id,
                schema,
                layout,
                stride_by_split,
                data_columns,
                split_ratios,
                subsequence_start_mode,
                split_method,
                seed,
            )
            post_split_str = f"{process_id}-{i}"

            for group, split in sequences.items():
                split_path = split_paths[group]
                split_path_batch_seq = split_path.replace(
                    f".{write_format}", f"-{post_split_str}.{write_format}"
                )
                split_path_batch_seq = insert_top_folder(
                    split_path_batch_seq, target_dir
                )

                if write_format == "csv":
                    write_data(split, split_path_batch_seq, "csv")
                elif write_format == "parquet":
                    write_data(split, split_path_batch_seq, "parquet")

                written_files[group].append(split_path_batch_seq)

        combine_multiprocessing_outputs(
            project_root,
            target_dir,
            len(split_paths),
            written_files,
            data_name_root,
            write_format,
            in_target_dir=True,
            post_split_str=f"{process_id}",
        )


@beartype
def extract_sequences(
    data: pl.DataFrame,
    schema: Any,
    layout: StoredWindowLayout,
    stride_for_split: int,
    columns: list[str],
    subsequence_start_mode: str,
) -> pl.DataFrame:
    """Extract long-format windows from grouped sequences."""
    if data.is_empty():
        return pl.DataFrame(schema=schema)

    raw_sequences = data.group_by("sequenceId", maintain_order=True).agg(
        [pl.col(c) for c in columns]
    )

    rows = []
    for in_row in raw_sequences.iter_rows(named=True):
        in_seq_lists_only = {col: in_row[col] for col in columns}

        subsequences, left_pad_lengths, subsequence_starts = extract_subsequences(
            in_seq_lists_only,
            layout.stored_context_width,
            stride_for_split,
            columns,
            subsequence_start_mode,
        )

        for subsequence_id in range(len(subsequences[columns[0]])):
            for col, subseqs in subsequences.items():
                row = [
                    in_row["sequenceId"],
                    subsequence_id,
                    int(subsequence_starts[subsequence_id])
                    - left_pad_lengths[subsequence_id],
                    left_pad_lengths[subsequence_id],
                    col,
                ] + subseqs[subsequence_id]
                expected_row_length = 5 + layout.stored_context_width
                if len(row) != expected_row_length:
                    raise RuntimeError(
                        f"Row length mismatch. Expected {expected_row_length}, got {len(row)}. Row: {row}"
                    )
                rows.append(row)

    sequences = pl.DataFrame(
        rows,
        schema=schema,
        orient="row",
    )
    return sequences


@beartype
def get_subsequence_starts(
    in_context_length: int,
    stored_context_width: int,
    stride_for_split: int,
    subsequence_start_mode: str,
) -> np.ndarray:
    """Return window start indices for distribute/exact modes."""
    if subsequence_start_mode not in ["distribute", "exact"]:
        raise ValueError(
            f"subsequence_start_mode must be 'distribute' or 'exact', got '{subsequence_start_mode}'"
        )

    if subsequence_start_mode == "distribute":
        last_available_start = in_context_length - stored_context_width
        raw_starts = np.arange(
            0, last_available_start + stride_for_split, stride_for_split
        )
        num_subsequences = len(raw_starts)

        starts = np.linspace(0, last_available_start, num_subsequences, dtype=int)

        return np.unique(starts)

    if subsequence_start_mode == "exact":
        if (in_context_length - stored_context_width) % stride_for_split != 0:
            raise ValueError(
                f"'exact' mode requires sequence length alignment, i.e. if: (in_context_length - stored_context_width) % stride_for_split == 0, {in_context_length = }, {stored_context_width = }, {stride_for_split = }"
            )
        last_possible_start = in_context_length - stored_context_width
        return np.arange(0, last_possible_start + 1, stride_for_split)
    return np.array([])


@beartype
def extract_subsequences(
    in_seq: dict[str, list],
    stored_context_width: int,
    stride_for_split: int,
    columns: list[str],
    subsequence_start_mode: str,
) -> tuple[dict[str, list[list[Union[float, int]]]], list[int], np.ndarray]:
    """Extract padded windows plus left-pad lengths from one sequence."""
    in_seq_len = len(in_seq[columns[0]])
    pad_len = 0
    if in_seq_len < stored_context_width:
        pad_len = stored_context_width - in_seq_len
        in_seq = {col: ([0] * pad_len) + in_seq[col] for col in columns}
    in_context_length = len(in_seq[columns[0]])

    subsequence_starts = get_subsequence_starts(
        in_context_length,
        stored_context_width,
        stride_for_split,
        subsequence_start_mode,
    )
    subsequence_starts_diff = subsequence_starts[1:] - subsequence_starts[:-1]
    if not np.all(subsequence_starts_diff <= stride_for_split):
        raise ValueError(
            f"Diff of {subsequence_starts = }, {subsequence_starts_diff = } larger than {stride_for_split = }"
        )

    result = {
        col: [
            list(in_seq[col][i : i + stored_context_width]) for i in subsequence_starts
        ]
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
    n_files_max = max(n_files.values()) if n_files else 1
    pad_width = len(str(n_files_max - 1))
    for split in range(n_splits):
        files_for_split = [
            os.path.join(
                project_root,
                "data",
                target_dir,
                f"{dataset_name}-{process_id}-{str(file_index).zfill(pad_width)}-split{split}.{write_format}",
            )
            for process_id in range(n_processes)
            for file_index in range(n_files[process_id])
        ]
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
            writer.write_table(pq.read_table(file, schema=schema))
