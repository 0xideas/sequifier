import glob
import hashlib
import os
import random
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Union

import numpy as np
import polars as pl
import torch
from beartype import BeartypeConf, BeartypeStrategy, beartype
from loguru import logger
from pydantic import ValidationError
from torch import Tensor

from sequifier.objectives import (
    ALLOWED_OBJECTIVE_NAMES,
    OBJECTIVE_NAME_MESSAGE,
    get_objective_class,
)
from sequifier.special_tokens import SPECIAL_TOKEN_IDS

PANDAS_TO_TORCH_TYPES = {
    "Float64": torch.float64,
    "float64": torch.float64,
    "Float32": torch.float32,
    "float32": torch.float32,
    "Float16": torch.float16,
    "float16": torch.float16,
    "Int64": torch.int64,
    "int64": torch.int64,
    "Int32": torch.int32,
    "int32": torch.int32,
    "Int16": torch.int16,
    "int16": torch.int16,
    "Int8": torch.int8,
    "int8": torch.int8,
    "UInt64": torch.uint64,
    "uint64": torch.uint64,
    "UInt32": torch.uint32,
    "uint32": torch.uint32,
    "UInt16": torch.uint16,
    "uint16": torch.uint16,
    "UInt8": torch.uint8,
    "uint8": torch.uint8,
}

POLARS_NUMERIC_DTYPES = {
    "Float64": pl.Float64,
    "Float32": pl.Float32,
    "Float16": pl.Float16,
    "Int64": pl.Int64,
    "Int32": pl.Int32,
    "Int16": pl.Int16,
    "Int8": pl.Int8,
    "UInt64": pl.UInt64,
    "UInt32": pl.UInt32,
    "UInt16": pl.UInt16,
    "UInt8": pl.UInt8,
}

POLARS_NUMERIC_DTYPE_ALIASES = {
    alias: canonical
    for canonical in POLARS_NUMERIC_DTYPES
    for alias in (canonical, canonical.lower())
}

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


@beartype
def canonicalize_polars_dtype_name(dtype_name: str) -> str:
    dtype_name = dtype_name.strip()
    if dtype_name not in POLARS_NUMERIC_DTYPE_ALIASES:
        raise ValueError(
            f"Unsupported column type '{dtype_name}'. "
            f"Supported types are: {sorted(POLARS_NUMERIC_DTYPES)}"
        )
    return POLARS_NUMERIC_DTYPE_ALIASES[dtype_name]


@beartype
def polars_dtype_from_name(dtype_name: str) -> Any:
    return POLARS_NUMERIC_DTYPES[canonicalize_polars_dtype_name(dtype_name)]


@beartype
def assign_sequence_to_split(
    sequence_id: int, split_ratios: list[float], seed: int
) -> int:
    """Deterministically assign one sequenceId to a split index."""
    digest = hashlib.sha256(f"{seed}:{sequence_id}".encode("utf-8")).digest()
    hash_value = int.from_bytes(digest[:8], byteorder="big", signed=False) / 2**64
    split_index = int(
        np.searchsorted(np.cumsum(split_ratios), hash_value, side="right")
    )
    return min(split_index, len(split_ratios) - 1)


@beartype
def is_float_dtype_name(dtype_name: str) -> bool:
    return canonicalize_polars_dtype_name(dtype_name).startswith("Float")


@beartype
def is_integer_dtype_name(dtype_name: str) -> bool:
    canonical = canonicalize_polars_dtype_name(dtype_name)
    return canonical.startswith("Int") or canonical.startswith("UInt")


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
def resolve_unified_polars_numeric_dtype(column_types: dict[str, str]) -> Any:
    """Resolve one Polars dtype for long-format numeric sequence columns."""
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
        return _resolve_integer_sequence_type(integer_types)

    resolved_float = _highest_ranked_type(float_types, FLOAT_TYPE_ORDER)
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
            resolved_float = required_float

    return polars_dtype_from_name(resolved_float)


@dataclass(frozen=True)
class StoredWindowLayout:
    stored_context_width: int
    max_target_offset: int
    version: int

    def __post_init__(self) -> None:
        if self.stored_context_width < 1:
            raise ValueError("stored_context_width must be a positive integer")
        if self.max_target_offset < 0:
            raise ValueError("max_target_offset must be non-negative")
        if self.max_target_offset >= self.stored_context_width:
            raise ValueError(
                "max_target_offset must be smaller than stored_context_width"
            )


@dataclass(frozen=True)
class ModelWindowView:
    context_length: int
    objective: str
    target_offset: int

    def __post_init__(self) -> None:
        if self.context_length < 1:
            raise ValueError("context_length must be a positive integer")
        if self.objective not in ALLOWED_OBJECTIVE_NAMES:
            raise ValueError(
                f"Only {OBJECTIVE_NAME_MESSAGE} are allowed, found {self.objective}"
            )
        if self.target_offset < 0:
            raise ValueError("target_offset must be non-negative")
        get_objective_class(self.objective).validate_window_view(
            self.context_length, self.target_offset
        )


@dataclass(frozen=True)
class ResolvedWindowView:
    storage: StoredWindowLayout
    view: ModelWindowView
    required_width: int
    input_slice: slice
    target_slice: slice

    def build_masks(self, left_pad_lengths: Tensor) -> dict[str, Tensor]:
        """Build explicit input-attention and target-validity masks for this view."""
        return {
            "attention_valid_mask": build_valid_mask(
                left_pad_lengths, self.storage.stored_context_width, self.input_slice
            ),
            "target_valid_mask": build_valid_mask(
                left_pad_lengths, self.storage.stored_context_width, self.target_slice
            ),
        }


@dataclass(frozen=True)
class ModelWindowSamplingPlan:
    """Resolve logical model windows contained in one stored window."""

    resolved_view: ResolvedWindowView
    stride: Optional[int] = None

    def __post_init__(self) -> None:
        if self.stride is not None and self.stride < 1:
            raise ValueError("model_window_stride must be a positive integer")

    @property
    def legacy_single_window(self) -> bool:
        return self.stride is None

    @property
    def max_input_start(self) -> int:
        start = self.resolved_view.input_slice.start
        if start is None:
            raise ValueError("Resolved input slice must have a concrete start")
        return start

    @property
    def candidate_input_starts(self) -> Tensor:
        """Return chronological starts, anchored to include the rightmost view."""
        max_start = self.max_input_start
        if self.legacy_single_window:
            return torch.tensor([max_start], dtype=torch.int64)

        assert self.stride is not None
        first_start = max_start % self.stride
        return torch.arange(
            first_start,
            max_start + 1,
            self.stride,
            dtype=torch.int64,
        )

    def first_eligible_start_indices(self, left_pad_lengths: Tensor) -> Tensor:
        """Return the first candidate with at least one valid target position."""
        left_pad_lengths = left_pad_lengths.to(dtype=torch.int64, device="cpu")
        if self.legacy_single_window:
            return torch.zeros_like(left_pad_lengths)

        assert self.stride is not None
        first_candidate = self.max_input_start % self.stride
        candidate_count = self.max_input_start // self.stride + 1
        target_last_offset = (
            self.resolved_view.view.target_offset
            + self.resolved_view.view.context_length
            - 1
        )
        minimum_starts = left_pad_lengths - target_last_offset
        first_indices = torch.div(
            minimum_starts - first_candidate + self.stride - 1,
            self.stride,
            rounding_mode="floor",
        )
        return first_indices.clamp(0, candidate_count)

    def sample_counts(self, left_pad_lengths: Tensor) -> Tensor:
        """Return the number of usable logical samples in each stored row."""
        left_pad_lengths = left_pad_lengths.to(dtype=torch.int64, device="cpu")
        if self.legacy_single_window:
            return torch.ones_like(left_pad_lengths)

        assert self.stride is not None
        candidate_count = self.max_input_start // self.stride + 1
        first_indices = self.first_eligible_start_indices(left_pad_lengths)
        return (candidate_count - first_indices).clamp_min(0)

    def sample_count_for_left_pad(self, left_pad_length: int) -> int:
        count = self.sample_counts(torch.tensor([left_pad_length], dtype=torch.int64))
        return int(count.item())

    def sample_count_from_histogram(
        self,
        histogram: Mapping[Any, int],
    ) -> int:
        return sum(
            int(frequency) * self.sample_count_for_left_pad(int(left_pad_length))
            for left_pad_length, frequency in histogram.items()
        )

    def build_index(self, left_pad_lengths: Tensor) -> "WindowSampleIndex":
        return WindowSampleIndex(self, left_pad_lengths)

    def gather(
        self,
        tensor: Tensor,
        stored_row_indices: Tensor,
        input_starts: Tensor,
        *,
        target: bool = False,
    ) -> Tensor:
        """Gather input or target windows without materializing all overlaps."""
        stored_row_indices = stored_row_indices.to(dtype=torch.int64, device="cpu")
        input_starts = input_starts.to(dtype=torch.int64, device="cpu")
        relative_positions = torch.arange(
            self.resolved_view.view.context_length,
            dtype=torch.int64,
        )
        position_offset = self.resolved_view.view.target_offset if target else 0
        positions = (
            input_starts[:, None] + position_offset + relative_positions[None, :]
        )
        return tensor[stored_row_indices[:, None], positions]

    def build_masks(
        self,
        left_pad_lengths: Tensor,
        input_starts: Tensor,
    ) -> dict[str, Tensor]:
        """Build masks for model windows with different positions in storage."""
        left_pad_lengths = left_pad_lengths.to(dtype=torch.int64, device="cpu")
        input_starts = input_starts.to(dtype=torch.int64, device="cpu")
        relative_positions = torch.arange(
            self.resolved_view.view.context_length,
            dtype=torch.int64,
        )
        input_positions = input_starts[:, None] + relative_positions[None, :]
        target_positions = input_positions + self.resolved_view.view.target_offset
        return {
            "attention_valid_mask": input_positions >= left_pad_lengths[:, None],
            "target_valid_mask": target_positions >= left_pad_lengths[:, None],
        }


class WindowSampleIndex:
    """Compact logical-index mapping for variable per-row window counts."""

    def __init__(
        self,
        plan: ModelWindowSamplingPlan,
        left_pad_lengths: Tensor,
    ) -> None:
        self.plan = plan
        self.left_pad_lengths = left_pad_lengths.to(dtype=torch.int64, device="cpu")
        self.starts = plan.candidate_input_starts
        self.first_start_indices = plan.first_eligible_start_indices(
            self.left_pad_lengths
        )
        self.counts = plan.sample_counts(self.left_pad_lengths)
        self.cumulative_counts = torch.cumsum(self.counts, dim=0)

    def __len__(self) -> int:
        if self.cumulative_counts.numel() == 0:
            return 0
        return int(self.cumulative_counts[-1].item())

    def share_memory_(self) -> "WindowSampleIndex":
        for tensor in (
            self.left_pad_lengths,
            self.starts,
            self.first_start_indices,
            self.counts,
            self.cumulative_counts,
        ):
            tensor.share_memory_()
        return self

    def resolve(self, logical_indices: Tensor) -> tuple[Tensor, Tensor]:
        logical_indices = torch.as_tensor(logical_indices, dtype=torch.int64)
        if logical_indices.numel() == 0:
            empty = torch.empty(0, dtype=torch.int64)
            return empty, empty
        if logical_indices.min().item() < 0 or logical_indices.max().item() >= len(
            self
        ):
            raise IndexError("Logical model-window sample index is out of range")

        stored_rows = torch.searchsorted(
            self.cumulative_counts,
            logical_indices,
            right=True,
        )
        previous_counts = torch.where(
            stored_rows == 0,
            torch.zeros_like(stored_rows),
            self.cumulative_counts[stored_rows - 1],
        )
        local_indices = logical_indices - previous_counts
        start_indices = self.first_start_indices[stored_rows] + local_indices
        return stored_rows, self.starts[start_indices]


@beartype
def _right_aligned_slice(width: int, length: int, offset: int) -> slice:
    start = width - (length + offset)
    stop = width - offset
    return slice(start, stop)


@beartype
def resolve_window_view(
    storage: StoredWindowLayout, view: ModelWindowView
) -> ResolvedWindowView:
    if view.target_offset > storage.max_target_offset:
        raise ValueError(
            f"Model target_offset={view.target_offset} exceeds stored "
            f"max_target_offset={storage.max_target_offset}."
        )

    input_offset = storage.max_target_offset
    target_offset = storage.max_target_offset - view.target_offset
    required_width = view.context_length + max(input_offset, target_offset)
    if required_width > storage.stored_context_width:
        raise ValueError(
            f"Model view requires width {required_width}, but storage only has "
            f"stored_context_width={storage.stored_context_width}."
        )

    return ResolvedWindowView(
        storage=storage,
        view=view,
        required_width=required_width,
        input_slice=_right_aligned_slice(
            storage.stored_context_width, view.context_length, input_offset
        ),
        target_slice=_right_aligned_slice(
            storage.stored_context_width, view.context_length, target_offset
        ),
    )


@beartype
def resolve_window_sampling_plan(
    storage: StoredWindowLayout,
    view: ModelWindowView,
    model_window_stride: Optional[int],
) -> ModelWindowSamplingPlan:
    return ModelWindowSamplingPlan(
        resolved_view=resolve_window_view(storage, view),
        stride=model_window_stride,
    )


def configured_model_window_stride(config: Any) -> Optional[int]:
    """Read the optional stride from validated configs or legacy test doubles."""
    value = getattr(config, "model_window_stride", None)
    return value if isinstance(value, int) else None


@beartype
def validate_stored_window_width(tensor: Tensor, stored_context_width: int) -> None:
    if tensor.shape[1] != stored_context_width:
        raise ValueError(
            f"Stored window width {tensor.shape[1]} does not match "
            f"metadata stored_context_width={stored_context_width}."
        )


@beartype
def stored_window_layout_from_metadata(metadata: dict) -> StoredWindowLayout:
    return StoredWindowLayout(
        stored_context_width=int(metadata["stored_context_width"]),
        max_target_offset=int(metadata["max_target_offset"]),
        version=int(metadata["stored_window_layout_version"]),
    )


# Check an environment variable to see if we are in a testing context
IS_TESTING = os.environ.get("SEQUIFIER_TESTING", "0") == "1"

# O1 is the default type-checking strategy. O0 completely disables it.
current_strategy = BeartypeStrategy.O1 if IS_TESTING else BeartypeStrategy.O0

conditional_beartype = beartype(conf=BeartypeConf(strategy=current_strategy))


@beartype
def try_catch_excess_keys(
    config_path: str, PydanticClass: Any, config_values: dict[Any, Any]
):
    try:
        return PydanticClass(
            **{k: v for k, v in config_values.items() if k != "skip_metadata"}
        )
    except ValidationError as e:
        # Filter the errors to find only the "extra fields"
        extra_fields = [
            err["loc"][0] for err in e.errors() if err["type"] == "value_error.extra"
        ]

        if extra_fields:
            raise ValueError(
                f"Found {len(extra_fields)} unrecognized configuration keys: {extra_fields}"
            ) from None

        raise e


@beartype
def construct_index_maps(
    id_maps: Optional[dict[str, dict[Union[str, int], int]]],
    target_columns_index_map: list[str],
    map_to_id: Optional[bool],
) -> dict[str, dict[int, Union[str, int]]]:
    """Build index-to-ID maps, including reserved token labels."""
    index_map = {}
    if map_to_id is not None and map_to_id:
        if id_maps is None:
            raise ValueError("id_maps cannot be None when map_to_id is True")
        for target_column in target_columns_index_map:
            map_ = {v: k for k, v in id_maps[target_column].items()}
            val = next(iter(map_.values()))
            if not isinstance(val, (str, int)):
                raise TypeError(
                    f"Expected string or integer ID in map, got {type(val)}"
                )
            map_.update(SPECIAL_TOKEN_IDS.labels_by_id)
            index_map[target_column] = map_
    return index_map


@beartype
def read_data(
    path: str, read_format: str, columns: Optional[list[str]] = None
) -> pl.DataFrame:
    """Read CSV/Parquet into Polars."""
    if read_format == "csv":
        return pl.read_csv(path, separator=",")
    if read_format == "parquet":
        return pl.read_parquet(path, columns=columns)
    raise ValueError(f"Unsupported read format: {read_format}")


@beartype
def write_data(data: pl.DataFrame, path: str, write_format: str, **kwargs) -> None:
    """Write Polars/Pandas data as CSV or Parquet."""
    if isinstance(data, pl.DataFrame):
        if write_format == "csv":
            data.write_csv(path, **kwargs)
        elif write_format == "parquet":
            data.write_parquet(path)
        else:
            raise ValueError(
                f"Unsupported write format for Polars DataFrame: {write_format}"
            )
        return

    if write_format == "csv":
        data.to_csv(path, separator=",", index=False, **kwargs)
    elif write_format == "parquet":
        data.to_parquet(path)
    else:
        raise ValueError(f"Unsupported write format: {write_format}")


@beartype
def subset_to_input_columns(
    data: Union[pl.DataFrame, pl.LazyFrame], input_columns: list[str]
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """Keep long-format rows whose inputCol is selected."""
    if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        return data.filter(pl.col("inputCol").is_in(input_columns))

    column_filters = [
        (data["inputCol"].values == input_col) for input_col in input_columns
    ]
    filter_ = np.logical_or.reduce(column_filters)
    return data.loc[filter_, :]


@beartype
def numpy_to_pytorch(
    data: pl.DataFrame,
    column_types: dict[str, torch.dtype],
    all_columns: list[str],
    resolved_view: ResolvedWindowView,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Convert long-format Polars windows to tensors plus masks."""
    input_seq_cols = columns_from_slice(
        resolved_view.input_slice, resolved_view.storage.stored_context_width
    )
    target_seq_cols = columns_from_slice(
        resolved_view.target_slice, resolved_view.storage.stored_context_width
    )

    unified_tensors = {}

    for col_name in all_columns:
        input_tensor = torch.tensor(
            data.filter(pl.col("inputCol") == col_name)
            .select(input_seq_cols)
            .to_numpy(),
            dtype=column_types[col_name],
        )
        unified_tensors[col_name] = input_tensor

        target_tensor = torch.tensor(
            data.filter(pl.col("inputCol") == col_name)
            .select(target_seq_cols)
            .to_numpy(),
            dtype=column_types[col_name],
        )
        unified_tensors[f"{col_name}_target"] = target_tensor

    left_pad_lengths = get_left_pad_lengths_from_preprocessed_data(data)
    metadata = resolved_view.build_masks(left_pad_lengths)

    return unified_tensors, metadata


@beartype
def numpy_storage_to_pytorch(
    data: pl.DataFrame,
    column_types: dict[str, torch.dtype],
    all_columns: list[str],
    stored_context_width: int,
    sort_rows: bool = True,
) -> tuple[dict[str, Tensor], Tensor]:
    """Convert complete stored windows to tensors for virtual window sampling."""
    sequence_columns = columns_from_slice(
        slice(0, stored_context_width),
        stored_context_width,
    )
    tensors = {}
    for column_name in all_columns:
        column_data = data.filter(pl.col("inputCol") == column_name)
        if column_data.is_empty():
            raise ValueError(f"Column not found in preprocessed data: {column_name}")
        if sort_rows:
            column_data = column_data.sort(["sequenceId", "subsequenceId"])
        tensors[column_name] = torch.tensor(
            column_data.select(sequence_columns).to_numpy(),
            dtype=column_types[column_name],
        )

    if sort_rows:
        left_pad_lengths = get_left_pad_lengths_from_preprocessed_data(data)
    else:
        left_pad_values = (
            data.group_by(["sequenceId", "subsequenceId"], maintain_order=True)
            .agg(pl.col("leftPadLength").first().alias("leftPadLength"))
            .get_column("leftPadLength")
        )
        left_pad_lengths = torch.tensor(
            left_pad_values.to_numpy(),
            dtype=torch.int64,
        )

    return tensors, left_pad_lengths


@beartype
def build_valid_mask(
    left_pad_lengths: Tensor,
    full_length: int,
    view_slice: slice,
) -> Tensor:
    """Boolean mask from left-padding metadata."""

    full_positions = torch.arange(
        full_length, device=left_pad_lengths.device, dtype=left_pad_lengths.dtype
    )
    full_valid = full_positions[None, :] >= left_pad_lengths[:, None]

    return full_valid[:, view_slice]


@beartype
def columns_from_slice(view_slice: slice, stored_context_width: int) -> list[str]:
    if view_slice.start is None or view_slice.stop is None:
        raise ValueError("Resolved window slices must have concrete bounds")
    return [
        str(stored_context_width - 1 - i)
        for i in range(view_slice.start, view_slice.stop)
    ]


@beartype
def get_left_pad_lengths_from_preprocessed_data(data: pl.DataFrame) -> Tensor:
    """One leftPadLength per long-format subsequence."""
    if "leftPadLength" not in data.columns:
        raise ValueError(
            "Dataset layout v1 does not contain explicit padding metadata. "
            "Please re-run preprocessing."
        )
    assert {"sequenceId", "subsequenceId"}.issubset(data.columns)

    lengths = (
        data.group_by(["sequenceId", "subsequenceId"], maintain_order=True)
        .agg(pl.col("leftPadLength").first().alias("leftPadLength"))
        .sort(["sequenceId", "subsequenceId"])
        .get_column("leftPadLength")
    )

    return torch.tensor(lengths.to_numpy(), dtype=torch.int64)


@beartype
def normalize_path(path: str, project_root: str) -> str:
    """Return path rooted under project_root."""
    project_root_normalized = (project_root + os.sep).replace(os.sep + os.sep, os.sep)
    path2 = os.path.join(project_root, path.replace(project_root_normalized, ""))
    return path2


@beartype
def configure_logger(project_root: str, model_name: str, rank: Optional[int] = 0):
    """Configure console plus rank-scoped debug/info log files."""
    logger.remove()

    if rank == 0 or rank is None:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>",
            level="INFO",
        )

    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    rank_str = f"rank{rank}" if rank is not None else "rank0"

    file_2_path = os.path.join(log_dir, f"sequifier-{model_name}-{rank_str}-2.txt")
    logger.add(
        file_2_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        enqueue=True,
        mode="a",
    )

    file_3_path = os.path.join(log_dir, f"sequifier-{model_name}-{rank_str}-3.txt")
    logger.add(
        file_3_path,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        enqueue=True,
        mode="a",
    )
    return logger


@beartype
def configure_determinism(seed: int, strict: bool = False) -> None:
    """Enforces deterministic execution for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if strict:
        torch.use_deterministic_algorithms(True, warn_only=True)

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


@beartype
def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """String-to-torch dtype mapping."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
    }

    # Add float8 support if available in this PyTorch version
    if hasattr(torch, "float8_e4m3fn"):
        dtype_map["float8_e4m3fn"] = torch.float8_e4m3fn
    if hasattr(torch, "float8_e5m2"):
        dtype_map["float8_e5m2"] = torch.float8_e5m2

    if dtype_str not in dtype_map:
        raise ValueError(
            f"dtype '{dtype_str}' not supported or available. Options: {list(dtype_map.keys())}"
        )

    return dtype_map[dtype_str]


def get_best_model_path(
    project_root: str, run_name: str, model_type: str
) -> tuple[str, int]:
    """Return the highest-epoch exported best model path."""
    search_pattern = os.path.join(
        project_root, "models", f"sequifier-{run_name}-best-*.{model_type}"
    )

    matching_models = glob.glob(search_pattern)

    if not matching_models:
        raise FileNotFoundError(
            f"Could not find an exported 'best' model matching: {search_pattern}"
        )

    best_model_path = max(
        matching_models,
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("-")[-1]),
    )
    last_epoch = int(best_model_path.split("-")[-1].split(".")[0])
    return best_model_path, last_epoch


def get_last_training_batch_timedelta(
    model_name: str, rank: int, project_root: str = "."
) -> float:
    """Return seconds between the last two mid-epoch train log entries."""
    log_path = os.path.join(
        project_root, "logs", f"sequifier-{model_name}-rank{rank}-2.txt"
    )

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    train_log_pattern = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+\|.*?\[INFO\] Epoch.*?Batch"
    )

    timestamps = []

    with open(log_path, "r", encoding="utf-8") as file:
        for line in file:
            match = train_log_pattern.search(line)
            if match:
                timestamps.append(
                    datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                )

    if len(timestamps) < 2:
        raise ValueError(
            "Not enough mid-epoch training logs found in the file to calculate a timedelta."
        )

    t1, t2 = timestamps[-2], timestamps[-1]

    return (t2 - t1).total_seconds()
