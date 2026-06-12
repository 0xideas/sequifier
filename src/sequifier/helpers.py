import glob
import os
import random
import re
import sys
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

EXPLICIT_PADDING_MASK_FALLBACK_WARNING = (
    "Explicit padding mask not found. Falling back to value-based padding "
    "inference for real-valued data; leading 0.0 values may be treated as "
    "padding. Re-run preprocessing to generate explicit masks."
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
    """Constructs reverse index maps (int index to original ID).

    This function creates reverse mappings from the integer indices back to
    the original string or integer identifiers. It only performs this
    operation if `map_to_id` is True and `id_maps` is provided.

    Special reserved IDs are always decoded as their sentinel labels:
    0 maps to "[unknown]", 1 maps to "[other]", and 2 maps to "[mask]".

    Args:
        id_maps: A nested dictionary mapping column names to their
            respective ID-to-index maps (e.g.,
            `{'col_name': {'original_id': 1, ...}}`). Expected to be
            provided if `map_to_id` is True.
        target_columns_index_map: A list of column names for which to
            construct the reverse maps.
        map_to_id: A boolean flag. If True, the reverse maps are
            constructed. If False or None, an empty dictionary is returned.

    Returns:
        A dictionary where keys are column names from
        `target_columns_index_map` and values are the reverse maps
        (index-to-original-ID). Returns an empty dict if `map_to_id`
        is not True.

    Raises:
        AssertionError: If `map_to_id` is True but `id_maps` is None.
        AssertionError: If the values of a map are not consistently
            string or integer (excluding the added '0' key).
    """
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
    """Reads data from a CSV or Parquet file into a Polars DataFrame.

    Args:
        path: The file path to read from.
        read_format: The format of the file. Supported formats are
            "csv" and "parquet".
        columns: An optional list of column names to read. This argument
            is only used when `read_format` is "parquet".

    Returns:
        A Polars DataFrame containing the data from the file.

    Raises:
        ValueError: If `read_format` is not "csv" or "parquet".
    """
    if read_format == "csv":
        return pl.read_csv(path, separator=",")
    if read_format == "parquet":
        return pl.read_parquet(path, columns=columns)
    raise ValueError(f"Unsupported read format: {read_format}")


@beartype
def write_data(data: pl.DataFrame, path: str, write_format: str, **kwargs) -> None:
    """Writes a Polars (or Pandas) DataFrame to a CSV or Parquet file.

    This function detects the type of the input DataFrame.
    - For Polars DataFrames, it uses `.write_csv()` or `.write_parquet()`.
    - For other DataFrame types (presumably Pandas), it uses `.to_csv()`
      or `.to_parquet()`.

    Note: The type hint specifies `pl.DataFrame`, but the implementation
    includes a fallback path that suggests compatibility with Pandas
    DataFrames.

    Args:
        data: The Polars (or Pandas) DataFrame to write.
        path: The destination file path.
        write_format: The format to write. Supported formats are
            "csv" and "parquet".
        **kwargs: Additional keyword arguments passed to the underlying
            write function (e.g., `write_csv` for Polars, `to_csv` for
            Pandas).

    Returns:
        None.

    Raises:
        ValueError: If `write_format` is not "csv" or "parquet".
    """
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
    """Filters a DataFrame to rows where 'inputCol' is in a list of column_names.

    This function supports both Polars (DataFrame, LazyFrame) and Pandas
    DataFrames, dispatching to the appropriate filtering method.

    - For Polars objects, it uses `data.filter(pl.col("inputCol").is_in(...))`.
    - For other objects (presumably Pandas), it builds a numpy boolean
      mask and filters using `data.loc[...]`.

    Note: The type hint only specifies Polars objects, but the
    implementation includes a fallback path for Pandas-like objects.

    Args:
        data: The Polars (DataFrame, LazyFrame) or Pandas DataFrame to
            filter. It must contain a column named "inputCol".
        input_columns: A list of values. Rows will be kept if their
            value in "inputCol" is present in this list.

    Returns:
        A filtered DataFrame or LazyFrame of the same type as the input.
    """
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
    seq_length: int,
    data_offset: int,
    target_offset: int,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Converts a long-format Polars DataFrame to a dict of sequence tensors.

    This function assumes the input DataFrame `data` is in a long format
    where each row represents a sequence for a specific feature. It expects
    a column named "inputCol" that contains the feature name (e.g.,
    'price', 'volume') and other columns representing time steps (e.g.,
    "0", "1", ..., "L").

    It generates two tensors for each column in `all_columns`:
    1.  An "input" tensor (from time steps L down to 1).
    2.  A "target" tensor (from time steps L-1 down to 0).

    Example:
        For `seq_length = 3` and `all_columns = ['price']`, it will create:
        - 'price': Tensor from columns ["3", "2", "1"]
        - 'price_target': Tensor from columns ["2", "1", "0"]

    Args:
        data: The long-format Polars DataFrame. Must contain "inputCol"
            and columns named as strings of integers for time steps.
        column_types: A dictionary mapping feature names (from "inputCol")
            to their desired `torch.dtype`.
        all_columns: A list of all feature names (from "inputCol") to
            be processed and converted into tensors.
        seq_length: The total sequence length (L). This determines the
            column names for time steps (e.g., "0" to "L").

    Returns:
        A tuple of:
        - a dictionary mapping feature names to their corresponding PyTorch
          tensors. Target tensors are stored with a `_target` suffix
          (e.g., `{'price': <tensor>, 'price_target': <tensor>}`).
        - a metadata dictionary containing any explicit masks.
    """
    input_seq_cols = [
        str(c) for c in range(seq_length - 1 + data_offset, (-1 + data_offset), -1)
    ]
    target_seq_cols = [
        str(c) for c in range(seq_length - 1 + target_offset, (-1 + target_offset), -1)
    ]

    # We will create a unified dictionary
    unified_tensors = {}

    for col_name in all_columns:
        # Create the input sequence tensor (e.g., from t=1 to t=L)
        input_tensor = torch.tensor(
            data.filter(pl.col("inputCol") == col_name)
            .select(input_seq_cols)
            .to_numpy(),
            dtype=column_types[col_name],
        )
        unified_tensors[col_name] = input_tensor

        # Create the target sequence tensor (e.g., from t=0 to t=L-1)
        # We'll store it with a "_target" suffix to distinguish it
        target_tensor = torch.tensor(
            data.filter(pl.col("inputCol") == col_name)
            .select(target_seq_cols)
            .to_numpy(),
            dtype=column_types[col_name],
        )
        unified_tensors[f"{col_name}_target"] = target_tensor

    left_pad_lengths = get_left_pad_lengths_from_preprocessed_data(data)
    if left_pad_lengths is not None:
        metadata = generate_padding_masks(
            left_pad_lengths, seq_length, data_offset, target_offset
        )
    else:
        metadata = {}

    return unified_tensors, metadata


@beartype
def build_valid_mask(
    left_pad_lengths: Tensor,
    full_length: int,
    offset: int,
    seq_length: int,
) -> Tensor:
    """Builds a boolean validity mask from explicit left-padding metadata."""

    full_positions = torch.arange(
        full_length, device=left_pad_lengths.device, dtype=left_pad_lengths.dtype
    )
    full_valid = full_positions[None, :] >= left_pad_lengths[:, None]

    return full_valid[:, -(seq_length + offset) : (-offset if offset > 0 else None)]


@beartype
def get_left_pad_lengths_from_preprocessed_data(data: pl.DataFrame) -> Optional[Tensor]:
    """Extracts one leftPadLength value per long-format subsequence."""
    if "leftPadLength" not in data.columns:
        return None

    assert {"sequenceId", "subsequenceId"}.issubset(data.columns)

    lengths = (
        data.group_by(["sequenceId", "subsequenceId"], maintain_order=True)
        .agg(pl.col("leftPadLength").first().alias("leftPadLength"))
        .sort(["sequenceId", "subsequenceId"])
        .get_column("leftPadLength")
    )

    return torch.tensor(lengths.to_numpy(), dtype=torch.int64)


@beartype
def generate_padding_masks(
    left_pad_lengths: Tensor,
    seq_length: int,
    data_offset: int,
    target_offset: int,
) -> dict[str, Tensor]:
    """Generates explicit attention and target masks as a metadata dictionary."""
    full_length = seq_length + 1
    return {
        "attention_valid_mask": build_valid_mask(
            left_pad_lengths, full_length, data_offset, seq_length
        ),
        "target_valid_mask": build_valid_mask(
            left_pad_lengths, full_length, target_offset, seq_length
        ),
    }


@beartype
def normalize_path(path: str, project_root: str) -> str:
    """Normalizes a path to be relative to a project path, then joins them.

    This function ensures that a given `path` is correctly expressed as
    an absolute path rooted at `project_root`. It does this by first
    removing the `project_root` prefix from `path` (if it exists)
    and then joining the result back to `project_root`.

    This is useful for handling paths that might be provided as either
    relative (e.g., "data/file.txt") or absolute
    (e.g., "/abs/path/to/project/data/file.txt").

    Args:
        path: The path to normalize.
        project_root: The absolute path to the project's root directory.

    Returns:
        A normalized, absolute path.
    """
    project_root_normalized = (project_root + os.sep).replace(os.sep + os.sep, os.sep)
    path2 = os.path.join(project_root, path.replace(project_root_normalized, ""))
    return path2


@beartype
def configure_logger(project_root: str, model_name: str, rank: Optional[int] = 0):
    """Configures Loguru to replicate the legacy LogFile behavior.

    Legacy Behavior Mapping:
    1. Console: Only Rank 0 prints high-level info.
    2. File 2 (Detailed): Captures ALL logs (equivalent to old level 2).
    3. File 3 (Summary): Captures only HIGH importance logs (equivalent to old level 3).
    4. Formatting: Files contain raw messages only (no timestamp prefix).
    """
    # Clear default handler
    logger.remove()

    # 1. Console Handler (Rank 0 only, INFO/Level 3 and up)
    if rank == 0 or rank is None:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>",
            level="INFO",
        )

    # Determine paths
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    rank_str = f"rank{rank}" if rank is not None else "rank0"

    # 2. File 2 (Detailed/Debug) - Equivalent to old 'level=2'
    # Captures everything from DEBUG up.
    # Format is just {message} to match f.write(f"{string}\n")
    file_2_path = os.path.join(log_dir, f"sequifier-{model_name}-{rank_str}-2.txt")
    logger.add(
        file_2_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        enqueue=True,
        mode="a",
    )

    # 3. File 3 (Summary/Info)
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
    # 1. Set standard seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 2. Ensure deterministic behavior in CUDA/CuDNN
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if strict:
        # 3. Enforce deterministic algorithms in PyTorch (crucial for SDPA/FlashAttention)
        # This forces PyTorch to error out if a non-deterministic operation is used,
        # or select the deterministic version of a kernel (e.g. for Flash Attn).
        torch.use_deterministic_algorithms(True, warn_only=True)

        # 4. Set CuBLAS workspace (Required for deterministic algorithms with CUDA >= 10.2)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


@beartype
def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Converts a string to a torch dtype, supporting bfloat16 and fp8."""
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
    """
    Searches for the exported 'best' model file for a given run and returns its path and epoch.

    Args:
        project_root: The root directory of the project.
        run_name: The unique identifier for the hyperparameter search run.
        model_type: The extension of the exported model (e.g., 'onnx' or 'pt').

    Returns:
        A tuple containing:
            - The file path to the best model (str).
            - The actual epoch at which this model was saved (int).

    Raises:
        FileNotFoundError: If no matching model files are found.
    """
    search_pattern = os.path.join(
        project_root, "models", f"sequifier-{run_name}-best-*.{model_type}"
    )

    matching_models = glob.glob(search_pattern)

    if not matching_models:
        raise FileNotFoundError(
            f"Could not find an exported 'best' model matching: {search_pattern}"
        )

    # Find the file with the highest epoch number in its name
    best_model_path = max(
        matching_models,
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("-")[-1]),
    )
    last_epoch = int(best_model_path.split("-")[-1].split(".")[0])
    return best_model_path, last_epoch


def get_last_training_batch_timedelta(
    model_name: str, rank: int, project_root: str = "."
) -> float:
    """
    Reads the level 2 log file, finds the last two mid-epoch training logs,
    and returns the timedelta between them in seconds.
    """
    # Construct the path to the level 2 log file based on configure_logger()
    log_path = os.path.join(
        project_root, "logs", f"sequifier-{model_name}-rank{rank}-2.txt"
    )

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    # Regex to capture the timestamp of mid-epoch training batch logs
    # Matches lines like: "2026-05-26 15:15:39 | INFO | [INFO] Epoch   1 | Batch   10/... | Loss: ..."
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

    # Get the last two chronologically recorded batch timestamps
    t1, t2 = timestamps[-2], timestamps[-1]

    return (t2 - t1).total_seconds()


def apply_bert_masking(
    data_batch: Dict[str, torch.Tensor],
    targets_batch: Dict[str, torch.Tensor],
    metadata_batch: Optional[Dict[str, torch.Tensor]],
    config: Any,  # TrainConfig
    eval_seed: Optional[int] = None,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Applies BERT-style span corruption to the input data using custom distributions.
    Explicitly passes the boolean prediction mask via the metadata dictionary.
    """
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

    if eval_seed is not None:
        cpu_rng_state = torch.get_rng_state()
        if device.type == "cuda":
            gpu_rng_state = torch.cuda.get_rng_state(device)
        torch.manual_seed(eval_seed)

    # Calculate exact number of tokens to mask per sequence based on valid length
    masking_prob = config.training_spec.bert_spec.masking_probability
    budgets = (valid_mask.sum(dim=1) * masking_prob).long()

    bert_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

    # 2. Pre-sample lengths and start positions to avoid massive GPU-CPU sync overhead
    # We sample more spans than needed to account for overlaps
    max_spans = int(seq_len * masking_prob) + 10

    sampled_lengths = config.training_spec.bert_spec.span_masking.sample(
        (batch_size, max_spans), device=device
    )

    sampled_starts_pct = torch.rand((batch_size, max_spans), device=device)
    # 3. Span Masking Loop
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

        # Apply spans until budget is hit
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

    # 4. Create the replacement split masks (e.g., 80% Mask, 10% Random, 10% Identical)
    replacement_probs = torch.rand((batch_size, seq_len), device=device)

    p_masked = config.training_spec.bert_spec.replacement_distribution.masked
    p_random = config.training_spec.bert_spec.replacement_distribution.random

    mask_token_mask = bert_mask & (replacement_probs < p_masked)
    random_token_mask = (
        bert_mask
        & (replacement_probs >= p_masked)
        & (replacement_probs < (p_masked + p_random))
    )

    # 5. Apply corruption to data_batch
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

    # 6. Append explicit prediction and attention masks to metadata.
    metadata_batch["bert_mask"] = bert_mask
    metadata_batch["attention_valid_mask"] = valid_mask.detach()

    if eval_seed is not None:
        torch.set_rng_state(cpu_rng_state)  # type: ignore
        if device.type == "cuda":
            torch.cuda.set_rng_state(gpu_rng_state, device)  # type: ignore
    return data_batch, targets_batch, metadata_batch
