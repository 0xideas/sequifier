import glob
import hashlib
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Union

import numpy as np
import polars as pl
import torch
from beartype import BeartypeConf, BeartypeStrategy, beartype
from loguru import logger
from pydantic import ValidationError
from torch import Tensor

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
        if self.objective not in {"causal", "bert"}:
            raise ValueError(
                f"Only 'causal' and 'bert' are allowed, found {self.objective}"
            )
        if self.target_offset < 0:
            raise ValueError("target_offset must be non-negative")
        if self.objective == "bert" and self.target_offset != 0:
            raise ValueError("BERT views require target_offset=0")
        if self.objective == "causal" and self.target_offset < 1:
            raise ValueError("Causal views require target_offset >= 1")


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


def _build_bert_span_mask(
    valid_mask: torch.Tensor,
    masking_probability: float,
    span_distribution: Any,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Construct exact-budget, non-overlapping BERT span masks."""
    valid_mask = valid_mask.bool()
    batch_size, seq_len = valid_mask.shape
    device = valid_mask.device

    valid_lengths = valid_mask.sum(dim=1, dtype=torch.long)
    budgets = (valid_lengths.to(torch.float32) * masking_probability).to(torch.long)

    max_spans = max(1, math.floor(seq_len * masking_probability) + 10)
    sampled_lengths = span_distribution.sample(
        (batch_size, max_spans),
        device=device,
        generator=generator,
    )
    sampled_lengths = sampled_lengths.to(torch.long).clamp_min_(1)

    used_before = sampled_lengths.cumsum(dim=1) - sampled_lengths
    remaining = (budgets[:, None] - used_before).clamp_min(0)
    span_lengths = torch.minimum(sampled_lengths, remaining)

    n_spans = (span_lengths > 0).sum(dim=1)
    total_gap_length = valid_lengths - budgets

    gap_slot = torch.arange(max_spans + 1, device=device)
    active_gap_slot = gap_slot[None, :] <= n_spans[:, None]

    uniform = torch.rand(
        (batch_size, max_spans + 1),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    uniform = uniform.clamp_min(torch.finfo(uniform.dtype).tiny)

    gap_weights = torch.where(
        active_gap_slot,
        -torch.log(uniform),
        torch.zeros_like(uniform),
    )
    cumulative_weights = gap_weights.cumsum(dim=1)
    weight_totals = cumulative_weights[:, -1:].clamp_min(
        torch.finfo(gap_weights.dtype).tiny
    )

    gap_edges = torch.floor(
        cumulative_weights
        / weight_totals
        * total_gap_length[:, None].to(gap_weights.dtype)
    ).to(torch.long)
    gap_edges = torch.where(
        gap_slot[None, :] >= n_spans[:, None],
        total_gap_length[:, None],
        gap_edges,
    )

    gaps = torch.diff(
        torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.long, device=device),
                gap_edges,
            ],
            dim=1,
        ),
        dim=1,
    )

    lengths_before = span_lengths.cumsum(dim=1) - span_lengths
    gaps_through_current = gaps[:, :max_spans].cumsum(dim=1)

    span_starts = lengths_before + gaps_through_current
    span_ends = span_starts + span_lengths

    compact_position = valid_mask.to(torch.long).cumsum(dim=1) - 1
    compact_position = compact_position.clamp_min(0)

    started_spans = torch.searchsorted(
        span_starts.contiguous(),
        compact_position.contiguous(),
        right=True,
    )
    ended_spans = torch.searchsorted(
        span_ends.contiguous(),
        compact_position.contiguous(),
        right=True,
    )

    return valid_mask & (started_spans > ended_spans)


def apply_bert_masking(
    data_batch: Dict[str, torch.Tensor],
    targets_batch: Dict[str, torch.Tensor],
    metadata_batch: Optional[Dict[str, torch.Tensor]],
    config: Any,  # TrainConfig
    eval_seed: Optional[int] = None,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Apply BERT span corruption and attach prediction masks."""
    if not metadata_batch or "attention_valid_mask" not in metadata_batch:
        raise ValueError("BERT masking requires metadata['attention_valid_mask']")

    valid_mask = metadata_batch["attention_valid_mask"].bool()
    batch_size, seq_len = valid_mask.shape
    device = valid_mask.device

    for target_name, target in targets_batch.items():
        if target.shape != valid_mask.shape:
            raise ValueError(
                f"BERT target {target_name!r} has shape {target.shape}; "
                f"expected {valid_mask.shape}"
            )

    generator: Optional[torch.Generator] = None
    if eval_seed is not None:
        seeded_generator = torch.Generator(device=device)
        seeded_generator.manual_seed(eval_seed)
        generator = seeded_generator

    bert_spec = config.training_spec.bert_spec
    if bert_spec is None:
        raise ValueError("bert_spec must be configured for BERT training")

    bert_mask = _build_bert_span_mask(
        valid_mask,
        bert_spec.masking_probability,
        bert_spec.span_masking,
        generator=generator,
    )

    replacement = bert_spec.replacement_distribution
    p_masked = replacement.masked
    p_random = replacement.random

    replacement_probs = torch.rand(
        (batch_size, seq_len),
        device=device,
        generator=generator,
    )

    mask_token_mask = bert_mask & (replacement_probs < p_masked)
    random_token_mask = (
        bert_mask
        & (replacement_probs >= p_masked)
        & (replacement_probs < (p_masked + p_random))
    )

    masked_data = dict(data_batch)

    for col, tensor in data_batch.items():
        if col in config.categorical_columns:
            output = tensor.clone()

            if p_masked > 0.0:
                output.masked_fill_(mask_token_mask, SPECIAL_TOKEN_IDS.mask)

            if p_random > 0.0:
                random_tokens = torch.randint(
                    low=SPECIAL_TOKEN_IDS.user_start,
                    high=config.n_classes[col],
                    size=tensor.shape,
                    device=device,
                    dtype=tensor.dtype,
                    generator=generator,
                )
                output[random_token_mask] = random_tokens[random_token_mask]

            masked_data[col] = output

        elif col in config.real_columns:
            output = tensor.clone()

            if p_masked > 0.0:
                output.masked_fill_(mask_token_mask, 0.0)

            if p_random > 0.0:
                random_noise = torch.randn(
                    tensor.shape,
                    device=device,
                    dtype=tensor.dtype,
                    generator=generator,
                )
                output[random_token_mask] = random_noise[random_token_mask]

            masked_data[col] = output

    detached_targets = {col: tensor.detach() for col, tensor in targets_batch.items()}
    output_metadata = {key: tensor.detach() for key, tensor in metadata_batch.items()}
    output_metadata["bert_mask"] = bert_mask
    output_metadata["attention_valid_mask"] = valid_mask

    return masked_data, detached_targets, output_metadata
