import glob
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
    "Float64": torch.float32,
    "float64": torch.float32,
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
    "UInt64": torch.int64,
    "uint64": torch.int64,
    "UInt32": torch.int64,
    "uint32": torch.int64,
    "UInt16": torch.int32,
    "uint16": torch.int32,
    "UInt8": torch.int16,
    "uint8": torch.int16,
}


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


def apply_bert_masking(
    data_batch: Dict[str, torch.Tensor],
    targets_batch: Dict[str, torch.Tensor],
    metadata_batch: Optional[Dict[str, torch.Tensor]],
    config: Any,  # TrainConfig
    eval_seed: Optional[int] = None,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Apply BERT span corruption and attach prediction masks."""
    data_batch = {k: tensor.clone() for k, tensor in data_batch.items()}
    targets_batch = {k: tensor.clone().detach() for k, tensor in targets_batch.items()}
    metadata_batch = (
        {k: tensor.clone().detach() for k, tensor in metadata_batch.items()}
        if metadata_batch
        else {}
    )

    valid_mask = metadata_batch["attention_valid_mask"].bool()

    batch_size, seq_len = valid_mask.shape
    device = valid_mask.device
    for target_name, target in targets_batch.items():
        if target.shape != valid_mask.shape:
            raise ValueError(
                f"BERT target {target_name!r} has shape {target.shape}; "
                f"expected {valid_mask.shape}"
            )

    if eval_seed is not None:
        cpu_rng_state = torch.get_rng_state()
        if device.type == "cuda":
            gpu_rng_state = torch.cuda.get_rng_state(device)
        torch.manual_seed(eval_seed)

    masking_prob = config.training_spec.bert_spec.masking_probability
    budgets = (valid_mask.sum(dim=1) * masking_prob).long()

    bert_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

    max_spans = int(seq_len * masking_prob) + 10

    sampled_lengths = config.training_spec.bert_spec.span_masking.sample(
        (batch_size, max_spans), device=device
    )

    sampled_starts_pct = torch.rand((batch_size, max_spans), device=device)
    for i in range(batch_size):
        budget = budgets[i].item()
        valid_positions = valid_mask[i].nonzero(as_tuple=True)[0]
        valid_len = len(valid_positions)

        if budget < 1 or valid_len < 1:
            continue

        sampled_starts = (sampled_starts_pct[i] * valid_len).long().tolist()

        sampled_lengths_list = sampled_lengths[i].tolist()

        current_masked = 0
        span_idx = 0

        while current_masked < budget and span_idx < max_spans:
            span_len = sampled_lengths_list[span_idx]
            start_idx = sampled_starts[span_idx]

            end_idx = min(start_idx + span_len, valid_len)
            span_positions = valid_positions[start_idx:end_idx]
            if len(span_positions) == 0:
                span_idx += 1
                continue

            unmasked_positions = span_positions[~bert_mask[i, span_positions]]
            allowance = budget - current_masked
            if len(unmasked_positions) > allowance:
                unmasked_positions = unmasked_positions[:allowance]

            bert_mask[i, unmasked_positions] = True
            current_masked = bert_mask[i].sum().item()
            span_idx += 1

        # Fallback: if heavy overlaps exhausted our max_spans, uniformly mask the remainder
        if current_masked < budget:
            remaining = budget - current_masked
            unmasked_valid = (valid_mask[i] & ~bert_mask[i]).nonzero(as_tuple=True)[0]
            if len(unmasked_valid) > 0:
                idx = torch.randperm(len(unmasked_valid), device=device)[:remaining]
                bert_mask[i, unmasked_valid[idx]] = True

    replacement_probs = torch.rand((batch_size, seq_len), device=device)

    p_masked = config.training_spec.bert_spec.replacement_distribution.masked
    p_random = config.training_spec.bert_spec.replacement_distribution.random

    mask_token_mask = bert_mask & (replacement_probs < p_masked)
    random_token_mask = (
        bert_mask
        & (replacement_probs >= p_masked)
        & (replacement_probs < (p_masked + p_random))
    )

    for col, tensor in data_batch.items():
        if col in config.categorical_columns:
            random_tokens = torch.randint(
                low=SPECIAL_TOKEN_IDS.user_start,
                high=config.n_classes[col],
                size=(batch_size, seq_len),
                device=device,
                dtype=tensor.dtype,
            )

            tensor[mask_token_mask] = SPECIAL_TOKEN_IDS.mask
            tensor[random_token_mask] = random_tokens[random_token_mask]

        elif col in config.real_columns:
            mask_val = 0.0
            random_noise = torch.randn(
                (batch_size, seq_len), device=device, dtype=tensor.dtype
            )

            tensor[mask_token_mask] = mask_val
            tensor[random_token_mask] = random_noise[random_token_mask]

    metadata_batch["bert_mask"] = bert_mask
    metadata_batch["attention_valid_mask"] = valid_mask.detach()

    if eval_seed is not None:
        torch.set_rng_state(cpu_rng_state)  # type: ignore
        if device.type == "cuda":
            torch.cuda.set_rng_state(gpu_rng_state, device)  # type: ignore
    return data_batch, targets_batch, metadata_batch
