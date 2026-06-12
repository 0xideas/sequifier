import json
import os
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import onnxruntime
import polars as pl
import torch
from beartype import beartype
from beartype.typing import Iterator
from loguru import logger

from sequifier.config.infer_config import InfererModel, load_inferer_config
from sequifier.helpers import (
    PANDAS_TO_TORCH_TYPES,
    configure_determinism,
    construct_index_maps,
    generate_padding_masks,
    normalize_path,
    numpy_to_pytorch,
    subset_to_input_columns,
    unpack_preprocessed_pt_tuple,
    write_data,
)
from sequifier.special_tokens import validate_special_token_ids
from sequifier.train import (
    infer_with_embedding_model,
    infer_with_generative_model,
    load_inference_model,
)


@beartype
def infer(args: Any, args_config: dict[str, Any]) -> None:
    """Runs the main inference pipeline.

    This function orchestrates the inference process. It loads the main
    inference configuration, retrieves necessary metadata like ID maps and
    column statistics from a `metadata_config` file (if required for mapping or
    normalization), and then delegates the core work to the `infer_worker`
    function.

    Args:
        args: Command-line arguments, typically from `argparse`. Expected
            to have attributes like `config_path` and `skip_metadata`.
        args_config: A dictionary of configuration overrides, often
            passed from the command line, that will be merged into the
            loaded configuration file.
    """
    logger.info("--- Starting Inference ---")
    config_path = (
        args.config_path if args.config_path is not None else "configs/infer.yaml"
    )

    skip_metadata = args_config.get("skip_metadata", False)
    config = load_inferer_config(config_path, args_config, skip_metadata)

    if config.map_to_id or (len(config.real_columns) > 0):
        if config.metadata_config_path is None:
            raise ValueError(
                "If you want to map to id, you need to provide a file path to a json that contains: {{'id_maps':{...}}} to metadata_config_path"
                "\nIf you have real columns in the data, you need to provide a json that contains: {{'selected_columns_statistics':{COL_NAME:{'std':..., 'mean':...}}}}"
            )
        with open(
            normalize_path(config.metadata_config_path, config.project_root), "r"
        ) as f:
            metadata_config = json.loads(f.read())
            validate_special_token_ids(
                metadata_config.get("special_token_ids"),
                source=f"metadata config '{config.metadata_config_path}'",
            )
            id_maps = metadata_config["id_maps"]
            selected_columns_statistics = metadata_config["selected_columns_statistics"]
    else:
        id_maps = None
        selected_columns_statistics = {}

    configure_determinism(config.seed, config.enforce_determinism)

    infer_worker(
        config, args_config, id_maps, selected_columns_statistics, (0.0, 100.0)
    )


@beartype
def load_pt_dataset(data_path: str, start_pct: float, end_pct: float) -> Iterator[Any]:
    """Lazily loads and yields data from .pt files in a directory.

    This function scans a directory for `.pt` files, sorts them, and then
    yields the contents of a specific slice of those files defined by a
    start and end percentage. This allows for processing large datasets
    in chunks without loading everything into memory.

    Args:
        data_path: The path to the folder containing the `.pt` files.
        start_pct: The starting percentage (0.0 to 100.0) of the file list
            to begin loading from.
        end_pct: The ending percentage (0.0 to 100.0) of the file list
            to stop loading at.

    Yields:
        Iterator: An iterator where each item is the data loaded from a
        single `.pt` file (e.g., using `torch.load`).
    """
    # Get all .pt files in the directory (not nested)
    pt_files = sorted(Path(data_path).glob("*.pt"))

    # Calculate slice indices
    total = len(pt_files)
    start_idx = int(total * start_pct / 100)
    end_idx = int(total * end_pct / 100)

    # Lazily load and yield data from files in range
    for pt_file in pt_files[start_idx:end_idx]:
        yield torch.load(pt_file, weights_only=False)


@beartype
def load_parquet_folder_dataset(
    data_path: str, start_pct: float, end_pct: float
) -> Iterator[Any]:
    """Lazily loads and yields data from long-format .parquet chunk files in a directory.

    This function scans a directory for `.parquet` files, sorts them, and then
    yields the contents of a specific slice of those files defined by a
    start and end percentage. This allows for processing large datasets
    in chunks without loading everything into memory.

    Args:
        data_path: The path to the folder containing the `.parquet` files.
        start_pct: The starting percentage (0.0 to 100.0) of the file list
            to begin loading from.
        end_pct: The ending percentage (0.0 to 100.0) of the file list
            to stop loading at.

    Yields:
        Iterator: An iterator where each item is a Polars DataFrame loaded from a
        single `.parquet` file.
    """
    parquet_files = sorted(Path(data_path).glob("*.parquet"))

    total = len(parquet_files)
    start_idx = int(total * start_pct / 100)
    end_idx = int(total * end_pct / 100)

    for parquet_file in parquet_files[start_idx:end_idx]:
        yield pl.read_parquet(parquet_file)


@beartype
def infer_worker(
    config: Any,
    args_config: dict[str, Any],
    id_maps: Optional[dict[str, dict[str | int, int]]],
    selected_columns_statistics: dict[str, dict[str, float]],
    percentage_limits: Optional[tuple[float, float]],
):
    """Core worker function that performs inference.

    This function handles the main workflow:
    1. Loads the dataset based on `config.read_format` (parquet, csv, or pt).
    2. Iterates over one or more model paths specified in the config.
    3. For each model, initializes an `Inferer` object with all necessary
       configurations, mappings, and statistics.
    4. Calls the appropriate inference function (`infer_generative` or
       `infer_embedding`) based on the `config.model_type`.
    5. Manages the data iterators and passes data chunks to the
       inference functions.

    Args:
        config: The fully resolved `InfererModel` configuration object.
        args_config: A dictionary of command-line arguments, passed to the
            `Inferer` for potential model loading overrides.
        id_maps: A nested dictionary mapping categorical column names to
            their value-to-index maps. `None` if `map_to_id` is False.
        selected_columns_statistics: A nested dictionary containing 'mean'
            and 'std' for real-valued columns used for normalization.
        percentage_limits: A tuple (start_pct, end_pct) used only when
            `config.read_format == "pt"` to slice the dataset.
    """
    logger.info(f"[INFO] Reading data from '{config.data_path}'...")

    is_folder_input = os.path.isdir(
        normalize_path(config.data_path, config.project_root)
    )
    # Step 1: Use Polars for data ingestion
    dataset = None
    if not is_folder_input:
        # Standalone Single-File Path Execution
        if config.read_format == "parquet":
            dataset = [pl.read_parquet(config.data_path)]
        elif config.read_format == "csv":
            dataset = [pl.read_csv(config.data_path)]

    model_paths = (
        config.model_path
        if isinstance(config.model_path, list)
        else [config.model_path]
    )
    for model_path in model_paths:
        if is_folder_input:
            if percentage_limits is None:
                raise ValueError(
                    "percentage_limits must be provided for folder-based read formats"
                )
            start_pct, end_pct = percentage_limits

            # Direct folders to their respective lazy loaders based on file format
            if config.read_format == "pt":
                dataset = load_pt_dataset(config.data_path, start_pct, end_pct)
            elif config.read_format == "parquet":
                dataset = load_parquet_folder_dataset(
                    config.data_path, start_pct, end_pct
                )

        if dataset is None:
            raise Exception(
                f"Unsupported input type or read format: {config.read_format}"
            )

        default_prediction_length = {"causal": 1, "bert": config.seq_length}
        prediction_length = (
            config.prediction_length
            if config.prediction_length is not None
            else default_prediction_length[config.training_objective]
        )

        inferer = Inferer(
            config.model_type,
            model_path,
            config.project_root,
            id_maps,
            selected_columns_statistics,
            config.map_to_id,
            config.categorical_columns,
            config.real_columns,
            config.input_columns,
            config.target_columns,
            config.target_column_types,
            config.sample_from_distribution_columns,
            config.infer_with_dropout,
            prediction_length,
            config.inference_batch_size,
            config.device,
            args_config=args_config,
            training_config_path=config.training_config_path,
        )

        column_types = {
            col: PANDAS_TO_TORCH_TYPES[config.column_types[col]]
            for col in config.column_types
        }

        model_id = os.path.split(model_path)[1].replace(
            f".{inferer.inference_model_type}", ""
        )

        logger.info(f"[INFO] Inferring for {model_id}")
        if config.model_type == "generative":
            infer_generative(config, inferer, model_id, dataset, column_types)
        if config.model_type == "embedding":
            infer_embedding(config, inferer, model_id, dataset, column_types)

    logger.info("--- Inference Complete ---")


def calculate_item_positions(
    start_positions: np.ndarray,
    seq_length: int,
    prediction_length: int,
    training_objective: str,
) -> np.ndarray:
    """
    Calculates absolute item positions for inference outputs based on the training objective.

    Args:
        start_positions: 1D array of base start positions for each sequence in the batch.
        seq_length: The length of the input sequence window.
        prediction_length: The total number of predicted tokens per sequence.
        training_objective: Either "causal" or "bert".

    Returns:
        A 1D array of absolute item positions mapped to every flattened prediction row.
    """
    if training_objective == "bert":
        # Anchor positions to the start of the input sequence and tile forwards
        base_positions = start_positions
        position_offsets = np.arange(0, prediction_length)
    else:
        # Anchor positions to the future token step and tile backwards
        base_positions = start_positions + seq_length
        position_offsets = np.arange(-prediction_length + 1, 1)

    # Repeat base anchors to match the number of predictions per sequence window
    repeated_bases = np.repeat(base_positions, prediction_length)
    # Tile the relative step offsets across all sequences in the batch chunk
    tiled_offsets = np.tile(position_offsets, len(start_positions))

    return repeated_bases + tiled_offsets


@beartype
def _flatten_bert_target_valid_mask(
    config: InfererModel,
    metadata: Optional[dict[str, Any]],
    prediction_length: int,
) -> Optional[np.ndarray]:
    if config.training_objective != "bert" or not metadata:
        return None
    if "target_valid_mask" not in metadata:
        return None

    valid_mask = metadata["target_valid_mask"]
    if isinstance(valid_mask, torch.Tensor):
        valid_mask = valid_mask.detach().cpu().numpy()
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if valid_mask.ndim != 2:
        raise ValueError(f"target_valid_mask must be 2D, got shape {valid_mask.shape}.")
    if valid_mask.shape[1] != prediction_length:
        raise ValueError(
            "target_valid_mask width must match prediction_length "
            f"(got {valid_mask.shape[1]} and {prediction_length})."
        )

    return valid_mask.reshape(-1)


@beartype
def _bert_reference_column(config: InfererModel, data_columns: set[str]) -> str:
    preferred_columns = (
        [col for col in config.target_columns if col in config.categorical_columns]
        + [col for col in config.input_columns if col in config.categorical_columns]
        + [col for col in config.target_columns if col in data_columns]
        + [col for col in config.input_columns if col in data_columns]
    )
    for column_name in preferred_columns:
        if column_name in data_columns:
            return column_name

    raise ValueError("Could not find a reference column for BERT padding metadata.")


@beartype
def _bert_target_valid_mask_from_preprocessed_data(
    config: InfererModel,
    data: pl.DataFrame,
    prediction_length: int,
) -> Optional[np.ndarray]:
    if config.training_objective != "bert":
        return None

    data_columns = set(data.get_column("inputCol").unique())
    column_name = _bert_reference_column(config, data_columns)
    reference_rows = data.filter(pl.col("inputCol") == column_name)

    if "leftPadLength" in reference_rows.columns:
        left_pad_lengths = torch.tensor(
            reference_rows.get_column("leftPadLength").to_numpy(), dtype=torch.int64
        )
        metadata = generate_padding_masks(
            left_pad_lengths,
            config.seq_length,
            data_offset=1,
            target_offset=1,
        )
        return _flatten_bert_target_valid_mask(config, metadata, prediction_length)

    return None


@beartype
def _apply_valid_prediction_mask(
    values: np.ndarray,
    valid_prediction_mask: Optional[np.ndarray],
    label: str,
) -> np.ndarray:
    values = np.asarray(values)
    if valid_prediction_mask is None:
        return values
    if values.shape[0] != valid_prediction_mask.shape[0]:
        raise ValueError(
            f"{label} has {values.shape[0]} rows, but target_valid_mask has "
            f"{valid_prediction_mask.shape[0]} rows."
        )
    return values[valid_prediction_mask]


@beartype
def _apply_valid_prediction_mask_to_dict(
    values: Optional[dict[str, np.ndarray]],
    valid_prediction_mask: Optional[np.ndarray],
    label: str,
) -> Optional[dict[str, np.ndarray]]:
    if values is None:
        return None
    return {
        key: _apply_valid_prediction_mask(
            value, valid_prediction_mask, f"{label}.{key}"
        )
        for key, value in values.items()
    }


@beartype
def infer_embedding(
    config: "InfererModel",
    inferer: "Inferer",
    model_id: str,
    dataset: Union[list[Any], Iterator[Any]],
    column_types: dict[str, torch.dtype],
) -> None:
    """Performs inference with an embedding model and saves the results.

    This function iterates through the provided dataset (which can be a list
    of DataFrames or an iterator of tensors). For each data chunk, it
    calls the appropriate function (`get_embeddings` or `get_embeddings_pt`)
    to generate embeddings. It then formats these embeddings into a
    Polars DataFrame, associating them with their `sequenceId`, `subsequenceId`,
    and absolute `itemPosition`, and writes the resulting DataFrame to the
    configured output path.

    Args:
        config: The `InfererModel` configuration object.
        inferer: The initialized `Inferer` instance.
        model_id: A string identifier for the model, used for naming
            output files.
        dataset: A list containing a Polars DataFrame (for parquet/csv) or
            an iterator of loaded PyTorch data (for .pt files).
        column_types: A dictionary mapping column names to their
            `torch.dtype`.
    """
    for data_id, data in enumerate(dataset):
        prediction_length = inferer.prediction_length
        valid_prediction_mask = None

        # Step 1: Get embeddings and base position/ID data
        is_folder_input = os.path.isdir(
            normalize_path(config.data_path, config.project_root)
        )

        if config.read_format in ["parquet", "csv"] and not is_folder_input:
            if config.input_columns is not None:
                data = subset_to_input_columns(data, config.input_columns)

            # Determine the number of input features
            n_input_cols = data.get_column("inputCol").n_unique()

            # Create a mask to select only one row per sequence
            mask = pl.arange(0, data.height, eager=True) % n_input_cols == 0

            embeddings = get_embeddings(config, inferer, data, column_types)
            valid_prediction_mask = _bert_target_valid_mask_from_preprocessed_data(
                config, data, prediction_length
            )

            sequence_ids_for_preds = data.get_column("sequenceId").filter(mask)
            subsequence_ids_for_preds = data.get_column("subsequenceId").filter(mask)
            item_positions_for_preds_base = (
                data.get_column("startItemPosition").filter(mask).to_numpy()
            )
        elif config.read_format == "parquet" and is_folder_input:
            # Folder-based Parquet chunk logic
            if config.input_columns is not None:
                data = subset_to_input_columns(data, config.input_columns)

            n_input_cols = data.get_column("inputCol").n_unique()
            mask = pl.arange(0, data.height, eager=True) % n_input_cols == 0

            embeddings = get_embeddings(config, inferer, data, column_types)
            valid_prediction_mask = _bert_target_valid_mask_from_preprocessed_data(
                config, data, prediction_length
            )

            sequence_ids_for_preds = (
                data.get_column("sequenceId").filter(mask).to_numpy()
            )
            subsequence_ids_for_preds = (
                data.get_column("subsequenceId").filter(mask).to_numpy()
            )
            item_positions_for_preds_base = (
                data.get_column("startItemPosition").filter(mask).to_numpy()
            )

        elif config.read_format == "pt":
            (
                sequences_dict,
                sequence_ids_tensor,
                subsequence_ids_tensor,
                start_positions_tensor,
                left_pad_lengths_tensor,
            ) = unpack_preprocessed_pt_tuple(data)
            metadata = {}
            if left_pad_lengths_tensor is not None:
                target_offset = 0 if config.training_objective == "causal" else 1
                metadata = generate_padding_masks(
                    left_pad_lengths_tensor,
                    config.seq_length,
                    data_offset=1,
                    target_offset=target_offset,
                )
            embeddings = get_embeddings_pt(
                config, inferer, sequences_dict, metadata=metadata
            )
            valid_prediction_mask = _flatten_bert_target_valid_mask(
                config, metadata, prediction_length
            )

            sequence_ids_for_preds = sequence_ids_tensor.numpy()
            subsequence_ids_for_preds = subsequence_ids_tensor.numpy()
            item_positions_for_preds_base = start_positions_tensor.numpy()

        else:
            raise Exception("impossible")

        # Step 2: Calculate absolute positions and repeat IDs
        # (e.g., for seq_len=50, inf_size=5, offsets are [45, 46, 47, 48, 49])
        base_offsets = np.arange(
            config.seq_length - prediction_length, config.seq_length
        )

        # Tile these offsets for each sample in the batch
        position_offsets_tiled = np.tile(
            base_offsets, len(item_positions_for_preds_base)
        )

        # Repeat the base start position for each of the N embedding outputs
        base_positions_repeated = np.repeat(
            item_positions_for_preds_base, prediction_length
        )

        # The final position is the start + the relative offset within the sequence
        final_positions = base_positions_repeated + position_offsets_tiled

        sequence_ids_repeated = np.repeat(sequence_ids_for_preds, prediction_length)
        subsequence_ids_repeated = np.repeat(
            subsequence_ids_for_preds, prediction_length
        )
        embeddings = _apply_valid_prediction_mask(
            embeddings, valid_prediction_mask, "embeddings"
        )
        final_positions = _apply_valid_prediction_mask(
            final_positions, valid_prediction_mask, "itemPosition"
        )
        sequence_ids_repeated = _apply_valid_prediction_mask(
            sequence_ids_repeated, valid_prediction_mask, "sequenceId"
        )
        subsequence_ids_repeated = _apply_valid_prediction_mask(
            subsequence_ids_repeated, valid_prediction_mask, "subsequenceId"
        )

        # Step 3: Build the final DataFrame
        embeddings_df = pl.DataFrame(
            {
                "sequenceId": sequence_ids_repeated,
                "subsequenceId": subsequence_ids_repeated,  # <-- ADDED THIS COLUMN
                "itemPosition": final_positions,
                **dict(
                    zip(
                        [str(v) for v in range(embeddings.shape[1])],
                        [embeddings[:, i] for i in range(embeddings.shape[1])],
                    )
                ),
            }
        )

        # Step 4: Save the output
        os.makedirs(
            os.path.join(config.project_root, "outputs", "embeddings"),
            exist_ok=True,
        )

        if not is_folder_input:
            file_name = f"{model_id}-embeddings.{config.write_format}"
        else:
            dirname = f"{model_id}-embeddings"
            file_name = os.path.join(
                dirname,
                f"{model_id}-{data_id}-embeddings.{config.write_format}",
            )

            dir_path = os.path.join(
                config.project_root, "outputs", "embeddings", dirname
            )
            os.makedirs(dir_path, exist_ok=True)

        embeddings_path = os.path.join(
            config.project_root, "outputs", "embeddings", file_name
        )
        logger.info(f"[INFO] Writing predictions to '{embeddings_path}'")
        write_data(
            embeddings_df,
            embeddings_path,
            config.write_format,
        )


def infer_generative(
    config: "InfererModel",
    inferer: "Inferer",
    model_id: str,
    dataset: Union[list[Any], Iterator[Any]],
    column_types: dict[str, torch.dtype],
):
    """Executes the generative inference pipeline and exports results to disk.

    This function processes the input dataset in chunks to accommodate large data
    volumes. It handles various input formats (standalone CSV/Parquet, folder-based
    Parquet, or PyTorch tensors) and routes the data to the appropriate inference
    logic (standard sequence prediction or step-by-step autoregression). After
    obtaining raw model outputs, it calculates aligned sequence IDs and absolute
    item positions, applies necessary post-processing (such as reverse-mapping
    categorical IDs and denormalizing real values), and writes the final
    probabilities and predictions to the configured output directory.

    Args:
        config: The inference configuration object dictating I/O paths,
            autoregression settings, and output formats.
        inferer: The initialized `Inferer` instance responsible for executing
            the underlying model logic.
        model_id: A string identifier for the current model, used to construct
            the names of the generated output files and directories.
        dataset: A list or iterator yielding data chunks, typically containing
            either Polars DataFrames or PyTorch tensor dictionaries.
        column_types: A dictionary mapping input column names to their expected
            `torch.dtype`.
    """
    for data_id, data in enumerate(dataset):
        valid_prediction_mask = None

        # Step 1: Adapt Data Subsetting (now works on Polars DF)
        is_folder_input = os.path.isdir(
            normalize_path(config.data_path, config.project_root)
        )

        if config.read_format in ["parquet", "csv"] and not is_folder_input:
            if config.input_columns is not None:
                data = subset_to_input_columns(data, config.input_columns)
            n_input_cols = data.get_column("inputCol").n_unique()
            if not config.autoregression:
                # For the non-autoregressive case, apply inference size logic
                probs, preds = get_probs_preds_from_df(
                    config, inferer, data, column_types
                )

                mask = pl.arange(0, data.height, eager=True) % n_input_cols == 0

                # Get base IDs and positions (shape: batch_size)
                sequence_ids_for_preds_base = data.get_column("sequenceId").filter(mask)
                item_positions_base_raw = (
                    data.get_column("startItemPosition").filter(mask).to_numpy()
                )
                prediction_length = inferer.prediction_length
                valid_prediction_mask = _bert_target_valid_mask_from_preprocessed_data(
                    config, data, prediction_length
                )

                # Expand IDs to match model output shape
                sequence_ids_for_preds = np.repeat(
                    sequence_ids_for_preds_base, prediction_length
                )

                # Invoke the unified positioning engine
                item_positions_for_preds = calculate_item_positions(
                    item_positions_base_raw,
                    config.seq_length,
                    prediction_length,
                    config.training_objective,
                )

            else:
                if inferer.prediction_length != 1:
                    raise ValueError(
                        f"prediction_length must be 1 for autoregression, got {inferer.prediction_length}"
                    )
                # Unpack the new third return value
                probs, preds, sequence_ids_for_preds, item_positions_for_preds = (
                    get_probs_preds_autoregression(
                        config, inferer, data, column_types, config.seq_length
                    )
                )
        elif config.read_format == "parquet" and is_folder_input:
            # Folder-based Parquet chunk logic
            if config.input_columns is not None:
                data = subset_to_input_columns(data, config.input_columns)
            n_input_cols = data.get_column("inputCol").n_unique()

            total_steps = (
                1
                if config.autoregression_total_steps is None
                else config.autoregression_total_steps
            )

            if total_steps == 1:
                probs, preds = get_probs_preds_from_df(
                    config, inferer, data, column_types
                )
                mask = pl.arange(0, data.height, eager=True) % n_input_cols == 0
                sequence_ids_for_preds_base = (
                    data.get_column("sequenceId").filter(mask).to_numpy()
                )
                item_positions_base_raw = (
                    data.get_column("startItemPosition").filter(mask).to_numpy()
                )
                prediction_length = inferer.prediction_length
                valid_prediction_mask = _bert_target_valid_mask_from_preprocessed_data(
                    config, data, prediction_length
                )

                sequence_ids_for_preds = np.repeat(
                    sequence_ids_for_preds_base, prediction_length
                )

                # Invoke the unified positioning engine
                item_positions_for_preds = calculate_item_positions(
                    item_positions_base_raw,
                    config.seq_length,
                    prediction_length,
                    config.training_objective,
                )
            else:
                if inferer.prediction_length != 1:
                    raise ValueError(
                        f"prediction_length must be 1 for autoregression, got {inferer.prediction_length}"
                    )

                probs, preds, sequence_ids_for_preds, item_positions_for_preds = (
                    get_probs_preds_autoregression(
                        config, inferer, data, column_types, config.seq_length
                    )
                )
        elif config.read_format == "pt":
            (
                sequences_dict,
                sequence_ids_tensor,
                _,
                start_positions_tensor,
                left_pad_lengths_tensor,
            ) = unpack_preprocessed_pt_tuple(data)
            total_steps = (
                1
                if config.autoregression_total_steps is None
                else config.autoregression_total_steps
            )

            sequences_dict = {
                key: tensor[:, :-1]
                for key, tensor in sequences_dict.items()
                if key in config.input_columns
            }

            metadata = {}
            if left_pad_lengths_tensor is not None:
                target_offset = 0 if config.training_objective == "causal" else 1
                metadata = generate_padding_masks(
                    left_pad_lengths_tensor,
                    config.seq_length,
                    data_offset=1,
                    target_offset=target_offset,
                )

            probs, preds = get_probs_preds_from_dict(
                config, inferer, sequences_dict, total_steps, metadata=metadata
            )

            prediction_length = inferer.prediction_length  # Get prediction_length
            valid_prediction_mask = _flatten_bert_target_valid_mask(
                config, metadata, prediction_length
            )

            if total_steps == 1:
                # Non-autoregressive path: Apply prediction_length logic
                sequence_ids_for_preds_base = sequence_ids_tensor.numpy()
                item_positions_base_raw = start_positions_tensor.numpy()

                sequence_ids_for_preds = np.repeat(
                    sequence_ids_for_preds_base, prediction_length
                )

                # Invoke the unified positioning engine
                item_positions_for_preds = calculate_item_positions(
                    item_positions_base_raw,
                    config.seq_length,
                    prediction_length,
                    config.training_objective,
                )

            else:
                sequence_ids_for_preds = np.repeat(
                    sequence_ids_tensor.numpy(), total_steps
                )
                item_position_boundaries = zip(
                    list(start_positions_tensor + config.seq_length),
                    list(start_positions_tensor + config.seq_length + total_steps),
                )
                item_positions_for_preds = np.concatenate(
                    [np.arange(start, end) for start, end in item_position_boundaries],
                    axis=0,
                )
        else:
            raise Exception("impossible")

        if inferer.map_to_id:
            for target_column, predictions in preds.items():
                if target_column in inferer.index_map:
                    preds[target_column] = np.array(
                        [inferer.index_map[target_column][i] for i in predictions]
                    )

        for target_column, predictions in preds.items():
            if inferer.target_column_types[target_column] == "real":
                preds[target_column] = inferer.invert_normalization(
                    predictions, target_column
                )

        sequence_ids_for_preds = _apply_valid_prediction_mask(
            sequence_ids_for_preds, valid_prediction_mask, "sequenceId"
        )
        item_positions_for_preds = _apply_valid_prediction_mask(
            item_positions_for_preds, valid_prediction_mask, "itemPosition"
        )
        preds = _apply_valid_prediction_mask_to_dict(
            preds, valid_prediction_mask, "preds"
        )
        probs = _apply_valid_prediction_mask_to_dict(
            probs, valid_prediction_mask, "probs"
        )

        os.makedirs(
            os.path.join(config.project_root, "outputs", "predictions"),
            exist_ok=True,
        )

        if config.output_probabilities:
            assert probs is not None
            os.makedirs(
                os.path.join(config.project_root, "outputs", "probabilities"),
                exist_ok=True,
            )

            for target_column in inferer.target_columns:
                if not is_folder_input:
                    file_name = f"{model_id}-{target_column}-probabilities.{config.write_format}"
                else:
                    dirname = f"{model_id}-{target_column}-probabilities"
                    file_name = os.path.join(
                        dirname,
                        f"{model_id}-{data_id}-probabilities.{config.write_format}",
                    )

                    dir_path = os.path.join(
                        config.project_root, "outputs", "probabilities", dirname
                    )
                    os.makedirs(dir_path, exist_ok=True)

                if inferer.target_column_types[target_column] == "categorical":
                    probabilities_path = os.path.join(
                        config.project_root, "outputs", "probabilities", file_name
                    )
                    logger.info(
                        f"[INFO] Writing probabilities to '{probabilities_path}'"
                    )
                    # Step 5: Finalize Output and I/O (write_data now handles Polars DF)
                    write_data(
                        pl.DataFrame(
                            probs[target_column],
                            schema=[
                                inferer.index_map[target_column][i]
                                for i in range(probs[target_column].shape[1])
                            ],
                        ),
                        probabilities_path,
                        config.write_format,
                    )

        n_input_cols = len(config.input_columns)

        assert preds is not None
        predictions = pl.DataFrame(
            {
                "sequenceId": sequence_ids_for_preds,
                "itemPosition": item_positions_for_preds,
                **{
                    target_column: preds[target_column].flatten()
                    for target_column in inferer.target_columns
                },
            }
        )

        if not is_folder_input:
            file_name = f"{model_id}-predictions.{config.write_format}"
        else:
            dirname = f"{model_id}-predictions"
            file_name = os.path.join(
                dirname, f"{model_id}-{data_id}-predictions.{config.write_format}"
            )
            dir_path = os.path.join(
                config.project_root, "outputs", "predictions", dirname
            )
            os.makedirs(dir_path, exist_ok=True)

        predictions_path = os.path.join(
            config.project_root, "outputs", "predictions", file_name
        )
        logger.info(f"[INFO] Writing predictions to '{predictions_path}'")
        write_data(
            predictions,
            predictions_path,
            config.write_format,
        )


@beartype
def get_embeddings_pt(
    config: Any,
    inferer: "Inferer",
    data: dict[str, torch.Tensor],
    metadata: Optional[dict[str, torch.Tensor]] = None,
) -> np.ndarray:
    """Generates embeddings from a batch of PyTorch tensor data.

    This function serves as a wrapper for `Inferer.infer_embedding` when
    the input data is already in PyTorch tensor format (from loading `.pt`
    files which contain sequences, targets, sequence_ids, subsequence_ids,
    and start_positions). It converts the tensor dictionary to a NumPy array
    dictionary before passing it to the inferer.

    Args:
        config: The `InfererModel` configuration object (unused, but
            kept for consistent function signature).
        inferer: The initialized `Inferer` instance.
        data: A dictionary mapping column/feature names to `torch.Tensor`s
              (the sequences part loaded from the .pt file).

    Returns:
        A NumPy array containing the computed embeddings for the batch.
    """
    X = {
        key: val[:, :-1].numpy()
        for key, val in data.items()
        if key in config.input_columns
    }
    metadata_np = (
        {key: val.numpy() for key, val in metadata.items()} if metadata else None
    )
    embeddings = inferer.infer_embedding(X, metadata=metadata_np)
    return embeddings


@beartype
def get_probs_preds_from_dict(
    config: Any,
    inferer: "Inferer",
    data: dict[str, torch.Tensor],
    total_steps: int = 1,
    metadata: Optional[dict[str, torch.Tensor]] = None,
) -> tuple[Optional[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Generates predictions from PyTorch tensor data, supporting autoregression.

    This function performs generative inference on a batch of PyTorch tensor
    data loaded from `.pt` files (which contain sequences, targets,
    sequence_ids, subsequence_ids, and start_positions). It implements an
    autoregressive loop:
    1. Runs inference on the initial data `X` (sequences).
    2. For each subsequent step:
       a. Creates the next input `X_next` by shifting the previous input
          `X` and appending the prediction from the last step.
       b. Runs inference on `X_next`.
    3. Collects and reshapes all predictions and probabilities from all
       steps into a single flat batch, ordered by original sample index, then by step.

    Args:
        config: The `InfererModel` configuration object, used to check
            `output_probabilities` and `input_columns`.
        inferer: The initialized `Inferer` instance.
        data: A dictionary mapping column/feature names to `torch.Tensor`s
              (the sequences part loaded from the .pt file).
        total_steps: The number of total autoregressive steps to
            perform. A value of 1 means simple, non-autoregressive
            inference.

    Returns:
        A tuple `(probs, preds)`:
            - `probs`: A dictionary mapping target columns to NumPy arrays
              of probabilities, ordered by sample index then step,
              or `None` if `config.output_probabilities` is False.
            - `preds`: A dictionary mapping target columns to NumPy arrays
              of final predictions, ordered by sample index then step.
    """

    target_cols = inferer.target_columns

    # 2. Initialize input and containers for storing results from all steps
    X = {
        key: tensor.numpy()
        for key, tensor in data.items()
        if key in config.input_columns
    }
    metadata_np = (
        {key: tensor.numpy() for key, tensor in metadata.items()} if metadata else None
    )
    all_probs_list = {col: [] for col in target_cols}
    all_preds_list = {col: [] for col in target_cols}

    # 3. Autoregressive loop
    metadata_for_step = metadata_np
    for i in range(total_steps):
        if config.output_probabilities:
            probs_for_step = inferer.infer_generative(
                X, return_probs=True, metadata=metadata_for_step
            )
            preds_for_step = inferer.infer_generative(None, probs_for_step)
            for col in target_cols:
                all_probs_list[col].append(probs_for_step[col])
        else:
            preds_for_step = inferer.infer_generative(
                X, return_probs=False, metadata=metadata_for_step
            )

        for col in target_cols:
            all_preds_list[col].append(preds_for_step[col])

        if i == (total_steps - 1):
            break

        X_next = {}
        for col in X.keys():
            shifted_input = X[col][:, 1:]

            new_value = preds_for_step[col].reshape(-1, 1).astype(shifted_input.dtype)

            X_next[col] = np.concatenate([shifted_input, new_value], axis=1)

        X = X_next
        if (
            metadata_for_step is not None
            and "attention_valid_mask" in metadata_for_step
        ):
            shifted_mask = metadata_for_step["attention_valid_mask"][:, 1:]
            appended_valid = np.ones(
                (shifted_mask.shape[0], 1), dtype=shifted_mask.dtype
            )
            metadata_for_step["attention_valid_mask"] = np.concatenate(
                [shifted_mask, appended_valid], axis=1
            )

    final_preds = {
        col: np.array(preds_list).T.reshape(-1, 1).flatten()
        for col, preds_list in all_preds_list.items()
    }

    if config.output_probabilities:
        final_probs = {
            col: np.array(probs_list)
            .transpose((1, 0, 2))
            .reshape(-1, probs_list[0].shape[1])
            for col, probs_list in all_probs_list.items()
        }
    else:
        final_probs = None

    return (final_probs, final_preds)


@beartype
def get_embeddings(
    config: Any,
    inferer: "Inferer",
    data: pl.DataFrame,
    column_types: dict[str, torch.dtype],
) -> np.ndarray:
    """Generates embeddings from a Polars DataFrame.

    This function converts a Polars DataFrame into the NumPy array dictionary
    format expected by the `Inferer`. It uses `numpy_to_pytorch` for the
    main conversion, then transforms the tensors to NumPy arrays before
    passing them to `inferer.infer_embedding`.

    Args:
        config: The `InfererModel` configuration object.
        inferer: The initialized `Inferer` instance.
        data: The input Polars DataFrame chunk.
        column_types: A dictionary mapping column names to `torch.dtype`.

    Returns:
        A NumPy array containing the computed embeddings for the batch.
    """
    all_columns = sorted(list(set(config.input_columns + config.target_columns)))
    target_offset = 0 if config.training_objective == "causal" else 1
    X, metadata = numpy_to_pytorch(
        data, column_types, all_columns, config.seq_length, 1, target_offset
    )
    X = {col: X_col.numpy() for col, X_col in X.items()}
    metadata_np = (
        {col: metadata_col.numpy() for col, metadata_col in metadata.items()}
        if metadata
        else None
    )
    del data

    embeddings = inferer.infer_embedding(X, metadata=metadata_np)

    return embeddings


@beartype
def get_probs_preds_from_df(
    config: Any,
    inferer: "Inferer",
    data: pl.DataFrame,
    column_types: dict[str, torch.dtype],
) -> tuple[Optional[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Generates predictions from a Polars DataFrame (non-autoregressive).

    This function converts a Polars DataFrame into the NumPy array dictionary
    format expected by the `Inferer`. It's used for standard,
    non-autoregressive generative inference.
    It calls `inferer.infer_generative` once and returns the
    probabilities (if requested) and predictions.

    Args:
        config: The `InfererModel` configuration object.
        inferer: The initialized `Inferer` instance.
        data: The input Polars DataFrame chunk.
        column_types: A dictionary mapping column names to `torch.dtype`.

    Returns:
        A tuple `(probs, preds)`:
            - `probs`: A dictionary mapping target columns to NumPy arrays
              of probabilities, or `None` if `config.output_probabilities`
              is False.
            - `preds`: A dictionary mapping target columns to NumPy arrays
              of final predictions.
    """
    all_columns = sorted(list(set(config.input_columns + config.target_columns)))

    target_offset = 0 if config.training_objective == "causal" else 1

    X, metadata = numpy_to_pytorch(
        data, column_types, all_columns, config.seq_length, 1, target_offset
    )
    X = {col: X_col.numpy() for col, X_col in X.items()}
    metadata_np = (
        {col: metadata_col.numpy() for col, metadata_col in metadata.items()}
        if metadata
        else None
    )
    del data

    if config.output_probabilities:
        probs = inferer.infer_generative(X, return_probs=True, metadata=metadata_np)
        preds = inferer.infer_generative(None, probs)
    else:
        probs = None
        preds = inferer.infer_generative(X, metadata=metadata_np)

    return (probs, preds)


@beartype
def fill_number(number: Union[int, float], max_length: int) -> str:
    """Pads a number with leading zeros to a specified string length.

    Used for creating sortable string keys (e.g., "001-001", "001-002").

    Args:
        number: The integer or float to format.
        max_length: The total desired length of the output string.

    Returns:
        A string representation of the number, padded with leading zeros.
    """
    number_str = str(number)
    return f"{'0' * (max_length - len(number_str))}{number_str}"


@beartype
def verify_variable_order(data: pl.DataFrame) -> None:
    """Verifies that the DataFrame is correctly sorted for autoregression.

    Checks two conditions:
    1. `sequenceId` is globally sorted in ascending order.
    2. `subsequenceId` is sorted in ascending order *within* each
       `sequenceId` group.

    Args:
        data: The Polars DataFrame to check.

    Raises:
        AssertionError: If `sequenceId` is not globally sorted or if
            `subsequenceId` is not sorted within `sequenceId` groups.
    """
    # Check if the entire 'sequenceId' column is sorted. This is a global property.
    is_globally_sorted = data.select(
        (pl.col("sequenceId").diff().fill_null(0) >= 0).all()
    ).item()
    if not is_globally_sorted:
        raise ValueError("sequenceId must be in ascending order for autoregression")

    # Check if 'subsequenceId' is sorted within each 'sequenceId' group.
    # This results in a boolean Series, on which we can call .all() directly.
    is_group_sorted = (
        data.select(
            (pl.col("subsequenceId").diff().fill_null(0) >= 0)
            .all()
            .over("sequenceId")
            .alias("is_sorted")
        )
        .get_column("is_sorted")
        .all()
    )

    if not is_group_sorted:
        raise ValueError("subsequenceId must be sorted within sequenceId groups")


@beartype
def get_probs_preds_autoregression(
    config: Any,
    inferer: "Inferer",
    data: pl.DataFrame,
    column_types: dict[str, torch.dtype],
    seq_length: int,
) -> tuple[
    Optional[dict[str, np.ndarray]], dict[str, np.ndarray], np.ndarray, np.ndarray
]:
    """Generates autoregressive predictions and aligns them with sequence IDs and positions.

    Extracts the initial sequence context from the sorted input DataFrame, maps it
    to PyTorch tensors, and executes step-by-step autoregressive inference.

    Args:
        config: Inference configuration object.
        inferer: Initialized `Inferer` instance.
        data: Input DataFrame, sorted globally by `sequenceId` and locally by `subsequenceId`.
        column_types: Mapping of input column names to their `torch.dtype`.
        seq_length: Length of the input sequence context.

    Returns:
        A tuple containing:
            - probs: Dict of probability arrays per target column (None if disabled).
            - preds: Dict of final prediction arrays per target column.
            - sequence_ids_for_preds: 1D array of sequence IDs matching the output shape.
            - item_positions_for_preds: 1D array of absolute item positions for each step.
    """
    verify_variable_order(data)

    distinct_cols = len(np.unique(data["inputCol"].to_numpy()))
    head_data_df = data.group_by("sequenceId", maintain_order=True).head(distinct_cols)

    aligned_sequence_ids = (
        head_data_df.get_column("sequenceId").unique(maintain_order=True).to_numpy()
    )

    aligned_start_positions = (
        head_data_df.group_by("sequenceId", maintain_order=True)
        .agg(pl.col("startItemPosition").max())
        .get_column("startItemPosition")
        .to_numpy()
        + seq_length
    )

    head_data, metadata = numpy_to_pytorch(
        head_data_df,
        column_types,
        config.input_columns,
        seq_length,
        data_offset=1,
        target_offset=0,
    )

    # Run the autoregressive PyTorch inference
    probs, preds = get_probs_preds_from_dict(
        config,
        inferer,
        head_data,
        total_steps=config.autoregression_total_steps,
        metadata=metadata,
    )

    # 4. Generate the final output arrays using the perfectly aligned bases
    item_positions_for_preds = np.concatenate(
        [
            np.arange(start_pos, start_pos + config.autoregression_total_steps)
            for start_pos in aligned_start_positions
        ],
        axis=0,
    )

    sequence_ids_for_preds = np.repeat(
        aligned_sequence_ids, config.autoregression_total_steps
    )

    return probs, preds, sequence_ids_for_preds, item_positions_for_preds


class Inferer:
    """A class for performing inference with a trained sequifier model.

    This class encapsulates the model (either ONNX session or PyTorch model),
    normalization statistics, ID mappings, and all configuration needed
    to run inference. It provides methods to handle batching, model-specific
    inference calls (PyTorch vs. ONNX), and post-processing
    (like inverting normalization).

    Attributes:
        model_type: 'generative' or 'embedding'.
        map_to_id: Whether to map integer predictions back to original IDs.
        selected_columns_statistics: Dict of 'mean' and 'std' for real columns.
        index_map: The inverse of `id_maps`, for mapping indices back to values.
        device: The device ('cuda' or 'cpu') for inference.
        target_columns: List of columns the model predicts.
        target_column_types: Dict mapping target columns to 'categorical' or 'real'.
        inference_model_type: 'onnx' or 'pt'.
        ort_session: `onnxruntime.InferenceSession` if using ONNX.
        inference_model: The loaded PyTorch model if using 'pt'.
    """

    @beartype
    def __init__(
        self,
        model_type: str,
        model_path: str,
        project_root: str,
        id_maps: Optional[dict[str, dict[Union[str, int], int]]],
        selected_columns_statistics: dict[str, dict[str, float]],
        map_to_id: bool,
        categorical_columns: list[str],
        real_columns: list[str],
        input_columns: Optional[list[str]],
        target_columns: list[str],
        target_column_types: dict[str, str],
        sample_from_distribution_columns: Optional[list[str]],
        infer_with_dropout: bool,
        prediction_length: int,
        inference_batch_size: int,
        device: str,
        args_config: dict[str, Any],
        training_config_path: str,
    ):
        """Initializes the Inferer.

        Args:
            model_type: The type of model to use for inference.
            model_path: The path to the trained model.
            project_root: The path to the sequifier project directory.
            id_maps: A dictionary of id maps for categorical columns.
            selected_columns_statistics: A dictionary of statistics for numerical columns.
            map_to_id: Whether to map the output to the original ids.
            categorical_columns: A list of categorical columns.
            real_columns: A list of real columns.
            selected_columns: A list of selected columns.
            target_columns: A list of target columns.
            target_column_types: A dictionary of target column types.
            sample_from_distribution_columns: A list of columns to sample from the distribution.
            infer_with_dropout: Whether to use dropout during inference.
            inference_batch_size: The batch size for inference.
            device: The device to use for inference.
            args_config: The command-line arguments.
            training_config_path: The path to the training configuration file.
        """
        self.model_type = model_type
        self.map_to_id = map_to_id
        self.selected_columns_statistics = selected_columns_statistics
        target_columns_index_map = [
            c for c in target_columns if target_column_types[c] == "categorical"
        ]
        self.index_map = construct_index_maps(
            id_maps, target_columns_index_map, map_to_id
        )

        self.device = device
        self.categorical_columns = categorical_columns
        self.real_columns = real_columns
        self.input_columns = input_columns
        self.target_columns = target_columns
        self.target_column_types = target_column_types
        self.sample_from_distribution_columns = sample_from_distribution_columns
        self.infer_with_dropout = infer_with_dropout
        self.prediction_length = prediction_length
        self.inference_batch_size = inference_batch_size

        self.inference_model_type = model_path.split(".")[-1]
        self.args_config = args_config
        self.training_config_path = training_config_path

        if self.inference_model_type == "onnx":
            execution_providers = [
                "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
            ]
            kwargs = {}
            if self.infer_with_dropout:
                kwargs["disabled_optimizers"] = ["EliminateDropout"]

                warnings.warn(
                    "For inference with onnx, 'infer_with_dropout==True' is only effective if 'export_with_dropout==True' in training"
                )

            self.ort_session = onnxruntime.InferenceSession(
                normalize_path(model_path, project_root),
                providers=execution_providers,
                **kwargs,
            )
        if self.inference_model_type == "pt":
            self.inference_model = load_inference_model(
                self.model_type,
                normalize_path(model_path, project_root),
                self.training_config_path,
                self.args_config,
                self.device,
                self.infer_with_dropout,
            )

    @beartype
    def invert_normalization(
        self, values: np.ndarray, target_column: str
    ) -> np.ndarray:
        """Inverts Z-score normalization for a given target column.

        Uses the 'mean' and 'std' stored in `self.selected_columns_statistics`
        to transform normalized values back to their original scale.

        Args:
            values: A NumPy array of normalized values.
            target_column: The name of the column whose statistics should be
                used for the inverse transformation.

        Returns:
            A NumPy array of values in their original scale.
        """
        std = self.selected_columns_statistics[target_column]["std"]
        mean = self.selected_columns_statistics[target_column]["mean"]
        return (values * (std - 1e-9)) + mean

    @beartype
    def infer_embedding(
        self,
        x: dict[str, np.ndarray],
        metadata: Optional[dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Performs inference with an embedding model.

        This is a high-level wrapper that calls
        `adjust_and_infer_embedding` to handle batching and model-specific
        logic.

        Args:
            x: A dictionary mapping feature names to NumPy arrays. All arrays
               must have the same first dimension (batch size).

        Returns:
            A 2D NumPy array of the resulting embeddings.
        """
        assert x is not None
        size = x[list(x.keys())[0]].shape[0]
        embedding = self.adjust_and_infer_embedding(x, size, metadata=metadata)

        return embedding

    @beartype
    def infer_generative(
        self,
        x: Optional[dict[str, np.ndarray]],
        probs: Optional[dict[str, np.ndarray]] = None,
        return_probs: bool = False,
        metadata: Optional[dict[str, np.ndarray]] = None,
    ) -> dict[str, np.ndarray]:
        """Performs generative inference, returning probabilities or predictions.

        This function orchestrates the generative inference process.
        1. If `probs` are not provided, it calls `adjust_and_infer_generative`
           to get the raw model output (logits or real values) using `x`.
        2. If `return_probs` is True:
           - It normalizes the logits for categorical columns to get
             probabilities (using `softmax`, implemented in `normalize`).
           - It returns a dictionary of probabilities (for categorical) and
             raw predicted values (for real).
        3. If `return_probs` is False (default):
           - It converts the model outputs (either from `x` or `probs`) into
             final predictions.
           - For categorical columns, it either takes the `argmax` or samples
             from the distribution (`sample_with_cumsum`).
           - For real columns, it returns the value as-is.

        Args:
            x: A dictionary mapping feature names to NumPy arrays. Required
               if `probs` is not provided.
            probs: An optional dictionary of probabilities/logits. If provided,
                   this skips the model inference step.
            return_probs: If True, returns normalized probabilities for
                categorical targets. If False, returns final class
                predictions (via argmax or sampling).

        Returns:
            A dictionary mapping target column names to NumPy arrays. The
            content of the arrays depends on `return_probs`.
        """
        if probs is None or (
            x is not None and len(set(x.keys()).difference(set(probs.keys()))) > 0
        ):  # type: ignore
            assert x is not None
            size = x[list(x.keys())[0]].shape[0]
            if (
                probs is not None
                and len(set(x.keys()).difference(set(probs.keys()))) > 0
            ):  # type: ignore
                assert x is not None
                warnings.warn(
                    f"Not all keys in x are in probs - {x.keys() = } != {probs.keys() = }. Full inference is executed."
                )

            outs = self.adjust_and_infer_generative(x, size, metadata=metadata)

            for target_column, target_outs in outs.items():
                if np.any(target_outs == np.inf):
                    raise ValueError(
                        f"Inference resulted in infinite values: {target_outs}"
                    )

            if return_probs:
                preds = {
                    target_column: outputs
                    for target_column, outputs in outs.items()
                    if self.target_column_types[target_column] != "categorical"
                }
                logits = {
                    target_column: outputs
                    for target_column, outputs in outs.items()
                    if self.target_column_types[target_column] == "categorical"
                }
                return {**preds, **normalize(logits)}
        else:
            outs = dict(probs)

        for target_column in self.target_columns:
            if self.target_column_types[target_column] == "categorical":
                if (
                    self.sample_from_distribution_columns is None
                    or target_column not in self.sample_from_distribution_columns
                ):
                    outs[target_column] = outs[target_column].argmax(1)
                else:
                    outs[target_column] = sample_with_cumsum(
                        outs[target_column], is_log_probs=(probs is None)
                    )

        return outs

    @beartype
    def adjust_and_infer_embedding(
        self,
        x: dict[str, np.ndarray],
        size: int,
        metadata: Optional[dict[str, np.ndarray]] = None,
    ):
        """Handles batching and backend-specific calls for embedding inference.

        This function prepares the input data `x` into batches using
        `prepare_inference_batches` and then calls the correct inference
        backend based on `self.inference_model_type` (.pt or .onnx).

        Args:
            x: The complete dictionary of input features (NumPy arrays).
            size: The total number of samples in `x`, used to truncate
                any padding added for batching.

        Returns:
            A NumPy array of embeddings, concatenated from all batches.
        """
        if self.inference_model_type == "onnx":
            assert x is not None
            x_adjusted = self.prepare_inference_batches(x, pad_to_batch_size=True)
            metadata_adjusted = (
                self.prepare_inference_batches(metadata, pad_to_batch_size=True)
                if metadata
                else [None] * len(x_adjusted)
            )
            inference_batch_embeddings = [
                self.infer_pure(x_sub, metadata_sub)[0]
                for x_sub, metadata_sub in zip(x_adjusted, metadata_adjusted)
            ]
            embeddings = np.concatenate(inference_batch_embeddings, axis=0)[:size]
        elif self.inference_model_type == "pt":
            x_adjusted = self.prepare_inference_batches(x, pad_to_batch_size=False)
            metadata_adjusted = (
                self.prepare_inference_batches(metadata, pad_to_batch_size=False)
                if metadata
                else None
            )
            embeddings = infer_with_embedding_model(
                self.inference_model,
                x_adjusted,
                self.device,
                size,
                self.target_columns,
                metadata=metadata_adjusted,
            )
        else:
            assert False, "not possible"
        return embeddings

    @beartype
    def adjust_and_infer_generative(
        self,
        x: dict[str, np.ndarray],
        size: int,
        metadata: Optional[dict[str, np.ndarray]] = None,
    ):
        """Handles batching and backend-specific calls for generative inference.

        This function prepares the input data `x` into batches using
        `prepare_inference_batches` and then calls the correct inference
        backend based on `self.inference_model_type` (.pt or .onnx).
        It aggregates the results from all batches.

        Args:
            x: The complete dictionary of input features (NumPy arrays).
            size: The total number of samples in `x`, used to truncate
                any padding added for batching.

        Returns:
            A dictionary mapping target column names to NumPy arrays of raw
            model outputs (logits or real values).
        """
        if self.inference_model_type == "onnx":
            assert x is not None
            x_adjusted = self.prepare_inference_batches(x, pad_to_batch_size=True)
            metadata_adjusted = (
                self.prepare_inference_batches(metadata, pad_to_batch_size=True)
                if metadata
                else [None] * len(x_adjusted)
            )
            out_subs = [
                dict(zip(self.target_columns, self.infer_pure(x_sub, metadata_sub)))
                for x_sub, metadata_sub in zip(x_adjusted, metadata_adjusted)
            ]
            outs = {
                target_column: np.concatenate(
                    [out_sub[target_column] for out_sub in out_subs], axis=0
                )[: size * self.prediction_length, :]
                for target_column in self.target_columns
            }
        elif self.inference_model_type == "pt":
            assert x is not None
            x_adjusted = self.prepare_inference_batches(x, pad_to_batch_size=False)
            metadata_adjusted = (
                self.prepare_inference_batches(metadata, pad_to_batch_size=False)
                if metadata
                else None
            )
            outs = infer_with_generative_model(
                self.inference_model,
                x_adjusted,
                self.device,
                size * self.prediction_length,
                self.target_columns,
                metadata=metadata_adjusted,
            )
        else:
            assert False
            outs = {}  # for type checking

        return outs

    @beartype
    def prepare_inference_batches(
        self, x: dict[str, np.ndarray], pad_to_batch_size: bool
    ) -> list[dict[str, np.ndarray]]:
        """Splits input data into batches for inference.

        This function takes a large dictionary of feature arrays and splits
        them into a list of smaller dictionaries (batches) of size
        `self.inference_batch_size`.

        Args:
            x: A dictionary of feature arrays.
            pad_to_batch_size: If True (for ONNX), the last batch will be
                padded up to `self.inference_batch_size` by repeating
                samples. If False (for PyTorch), the last batch may be
                smaller.

        Returns:
            A list of dictionaries, where each dictionary is a single batch
            ready for inference.
        """
        size = x[list(x.keys())[0]].shape[0]
        if size == self.inference_batch_size:
            return [x]
        elif size < self.inference_batch_size:
            if pad_to_batch_size:
                x_expanded = {
                    col: self.expand_to_batch_size(x_col) for col, x_col in x.items()
                }
                return [x_expanded]
            else:
                return [x]
        else:
            starts = range(0, size, self.inference_batch_size)
            ends = range(
                self.inference_batch_size,
                size + self.inference_batch_size,
                self.inference_batch_size,
            )
            xs = [
                {col: x_col[start:end, :] for col, x_col in x.items()}
                for start, end in zip(starts, ends)
            ]
            return xs

    @beartype
    def infer_pure(
        self,
        x: dict[str, np.ndarray],
        metadata: Optional[dict[str, np.ndarray]] = None,
    ) -> list[np.ndarray]:
        """Performs a single inference pass using the ONNX session.

        This function assumes `x` is already a single, correctly-sized
        batch. It formats the input dictionary to match the ONNX model's
        input names and executes `self.ort_session.run()`.

        Args:
            x: A dictionary of feature arrays for a single batch. This
               batch *must* be of size `self.inference_batch_size`.

        Returns:
            A list of NumPy arrays, representing the raw outputs from the
            ONNX model.
        """
        metadata = metadata or {}
        reference_shape = next(iter(x.values())).shape
        ort_inputs = {}
        for session_input in self.ort_session.get_inputs():
            input_name = session_input.name
            if input_name == "attention_valid_mask":
                value = metadata.get(input_name)
                if value is None:
                    value = np.ones(reference_shape[:2], dtype=np.bool_)
            elif input_name in metadata:
                value = metadata[input_name]
            elif input_name.endswith("_in") and input_name[:-3] in x:
                feature_column = input_name[:-3]
                value = x[feature_column]
            elif input_name in x:
                value = x[input_name]
            else:
                raise ValueError(
                    f"Could not map ONNX input '{input_name}' to a feature or metadata array."
                )

            ort_inputs[input_name] = self.expand_to_batch_size(value)

        ort_outs = self.ort_session.run(None, ort_inputs)
        return [
            oo.transpose(1, 0, 2).reshape(oo.shape[0] * oo.shape[1], oo.shape[2])
            for oo in ort_outs
        ]

    @beartype
    def expand_to_batch_size(self, x: np.ndarray) -> np.ndarray:
        """Pads a NumPy array to match `self.inference_batch_size`.

        Repeats samples from `x` until the array's first dimension
        is equal to `self.inference_batch_size`.

        Args:
            x: The input NumPy array to pad.

        Returns:
            A new NumPy array of size `self.inference_batch_size` in the
            first dimension.
        """
        repetitions = self.inference_batch_size // x.shape[0]
        filler = self.inference_batch_size % x.shape[0]
        return np.concatenate(([x] * repetitions) + [x[0:filler, :]], axis=0)


@beartype
def normalize(outs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Applies the softmax function to a dictionary of logits.

    Converts raw model logits for categorical columns into probabilities
    that sum to 1.

    Args:
        outs: A dictionary mapping target column names to NumPy arrays
              of logits.

    Returns:
        A dictionary mapping the same target column names to NumPy arrays
        of probabilities.
    """
    normalizer = {
        target_column: np.repeat(
            np.sum(np.exp(target_values), axis=1), target_values.shape[1]
        ).reshape(target_values.shape)
        for target_column, target_values in outs.items()
    }
    probs = {
        target_column: np.exp(target_values) / normalizer[target_column]
        for target_column, target_values in outs.items()
    }
    return probs


@beartype
def sample_with_cumsum(probs: np.ndarray, is_log_probs: bool = True) -> np.ndarray:
    """Samples from a probability distribution using the inverse CDF method.

    Takes an array of logits, computes the cumulative probability
    distribution, draws a random number `r` from [0, 1), and returns
    the index of the first class `i` where `cumsum[i] > r`.

    Args:
        probs: A 2D NumPy array of logits or normalized probabilities.
               Shape is (batch_size, num_classes).
        is_log_probs: Boolean flag indicating if the passed array are logits or
               probabilities
    Returns:
        A 1D NumPy array of shape (batch_size,) containing the sampled
        class indices.
    """
    if is_log_probs:
        cumulative_probs = np.cumsum(np.exp(probs), axis=1)
    else:
        cumulative_probs = np.cumsum(probs, axis=1)
    random_threshold = np.random.rand(cumulative_probs.shape[0], 1)
    random_threshold = np.repeat(random_threshold, probs.shape[1], axis=1)
    return (random_threshold < cumulative_probs).argmax(axis=1)
