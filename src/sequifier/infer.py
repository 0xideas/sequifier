import json
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np
import onnxruntime
import polars as pl
import torch
from beartype import beartype

from sequifier.config.infer_config import InfererModel, load_inferer_config
from sequifier.helpers import (
    PANDAS_TO_TORCH_TYPES,
    construct_index_maps,
    normalize_path,
    numpy_to_pytorch,
    subset_to_input_columns,
    write_data,
)
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
            to have attributes like `config_path` and `on_unprocessed`.
        args_config: A dictionary of configuration overrides, often
            passed from the command line, that will be merged into the
            loaded configuration file.
    """
    print("--- Starting Inference ---")
    config_path = (
        args.config_path if args.config_path is not None else "configs/infer.yaml"
    )

    on_unprocessed = args_config.get("on_unprocessed", False)
    config = load_inferer_config(config_path, args_config, on_unprocessed)

    if config.map_to_id or (len(config.real_columns) > 0):
        assert config.metadata_config_path is not None, (
            "If you want to map to id, you need to provide a file path to a json that contains: {{'id_maps':{...}}} to metadata_config_path"
            "\nIf you have real columns in the data, you need to provide a json that contains: {{'selected_columns_statistics':{COL_NAME:{'std':..., 'mean':...}}}}"
        )
        with open(
            normalize_path(config.metadata_config_path, config.project_path), "r"
        ) as f:
            metadata_config = json.loads(f.read())
            id_maps = metadata_config["id_maps"]
            selected_columns_statistics = metadata_config["selected_columns_statistics"]
    else:
        id_maps = None
        selected_columns_statistics = {}

    infer_worker(
        config, args_config, id_maps, selected_columns_statistics, (0.0, 100.0)
    )


@beartype
def load_pt_dataset(data_path: str, start_pct: float, end_pct: float) -> Iterator:
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
        yield torch.load(pt_file)


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
    print(f"[INFO] Reading data from '{config.data_path}'...")
    # Step 1: Use Polars for data ingestion
    dataset = None
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
        if config.read_format == "pt":
            assert percentage_limits is not None
            start_pct, end_pct = percentage_limits
            dataset = load_pt_dataset(config.data_path, start_pct, end_pct)

        if dataset is None:
            raise Exception(f"{config.read_format = } not in ['parquet', 'csv', 'pt']")

        inferer = Inferer(
            config.model_type,
            model_path,
            config.project_path,
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
            config.prediction_length,
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

        print(f"[INFO] Inferring for {model_id}")
        if config.model_type == "generative":
            infer_generative(config, inferer, model_id, dataset, column_types)
        if config.model_type == "embedding":
            infer_embedding(config, inferer, model_id, dataset, column_types)

    print("--- Inference Complete ---")


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

        # Step 1: Get embeddings and base position/ID data
        if config.read_format in ["parquet", "csv"]:
            if config.input_columns is not None:
                data = subset_to_input_columns(data, config.input_columns)

            # Determine the number of input features
            n_input_cols = data.get_column("inputCol").n_unique()

            # Create a mask to select only one row per sequence
            mask = pl.arange(0, data.height, eager=True) % n_input_cols == 0

            embeddings = get_embeddings(config, inferer, data, column_types)

            sequence_ids_for_preds = data.get_column("sequenceId").filter(mask)
            # --- ADDED THIS LINE ---
            subsequence_ids_for_preds = data.get_column("subsequenceId").filter(mask)
            item_positions_for_preds_base = (
                data.get_column("startItemPosition").filter(mask).to_numpy()
            )

        elif config.read_format == "pt":
            (
                sequences_dict,
                _,
                sequence_ids_tensor,
                subsequence_ids_tensor,
                start_positions_tensor,
            ) = data
            embeddings = get_embeddings_pt(config, inferer, sequences_dict)

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
            os.path.join(config.project_path, "outputs", "embeddings"),
            exist_ok=True,
        )

        if config.read_format in ["csv", "parquet"]:
            file_name = f"{model_id}-embeddings.{config.write_format}"
        else:
            dirname = f"{model_id}-embeddings"
            file_name = os.path.join(
                dirname,
                f"{model_id}-{data_id}-embeddings.{config.write_format}",
            )

            dir_path = os.path.join(
                config.project_path, "outputs", "embeddings", dirname
            )
            os.makedirs(dir_path, exist_ok=True)

        embeddings_path = os.path.join(
            config.project_path, "outputs", "embeddings", file_name
        )
        print(f"[INFO] Writing predictions to '{embeddings_path}'")
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
    """Performs inference with a generative model and saves the results.

    This function manages the generative inference workflow:
    1. Iterates through the dataset (chunks).
    2. Handles data preparation, including expanding data for autoregression
       if configured (`expand_data_by_autoregression`). It also calculates
       the corresponding `itemPosition` for each prediction.
    3. Calls the correct function to get probabilities and predictions
       based on data format and autoregression settings (e.g.,
       `get_probs_preds_autoregression`, `get_probs_preds_pt`).
    4. Post-processes predictions:
       - Maps integer predictions back to original IDs if `map_to_id` is True.
       - Inverts normalization for real-valued target columns.
    5. Saves probabilities to disk (if `config.output_probabilities` is True).
    6. Saves the final predictions to disk, formatted as a Polars DataFrame
       with `sequenceId`, `itemPosition`, and target columns.

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
        # Step 1: Adapt Data Subsetting (now works on Polars DF)
        if config.read_format in ["parquet", "csv"]:
            if config.input_columns is not None:
                data = subset_to_input_columns(data, config.input_columns)
            n_input_cols = data.get_column("inputCol").n_unique()
            if not config.autoregression:
                # For the non-autoregressive case, apply inference size logic
                probs, preds = get_probs_preds(config, inferer, data, column_types)

                mask = pl.arange(0, data.height, eager=True) % n_input_cols == 0

                # Get base IDs and positions (shape: batch_size)
                sequence_ids_for_preds_base = data.get_column("sequenceId").filter(mask)
                item_positions_for_preds_base = (
                    data.get_column("startItemPosition").filter(mask).to_numpy()
                    + config.seq_length
                )

                prediction_length = inferer.prediction_length

                # Expand IDs and positions to match model output shape (batch_size * prediction_length)
                sequence_ids_for_preds = np.repeat(
                    sequence_ids_for_preds_base, prediction_length
                )

                item_positions_repeated = np.repeat(
                    item_positions_for_preds_base, prediction_length
                )
                position_offsets = np.tile(
                    np.arange(-prediction_length + 1, 1),
                    len(item_positions_for_preds_base),
                )
                item_positions_for_preds = item_positions_repeated + position_offsets

            else:
                assert (
                    inferer.prediction_length == 1
                ), f"{inferer.prediction_length = } != 1, is not allowed for autoregressive inference"
                if config.autoregression_extra_steps is not None:
                    data = expand_data_by_autoregression(
                        data,
                        config.autoregression_extra_steps,
                        config.seq_length,
                    )
                mask = pl.arange(0, data.height, eager=True) % n_input_cols == 0
                item_positions_for_preds = (
                    data.get_column("startItemPosition").filter(mask).to_numpy()
                    + config.seq_length
                )
                # Unpack the new third return value
                probs, preds, sequence_ids_for_preds = get_probs_preds_autoregression(
                    config, inferer, data, column_types, config.seq_length
                )
        elif config.read_format == "pt":
            sequences_dict, _, sequence_ids_tensor, _, start_positions_tensor = data
            extra_steps = (
                0
                if config.autoregression_extra_steps is None
                else config.autoregression_extra_steps
            )

            # Pass prediction_length to get_probs_preds_pt
            probs, preds = get_probs_preds_pt(
                config, inferer, sequences_dict, extra_steps
            )

            prediction_length = inferer.prediction_length  # Get prediction_length

            if extra_steps == 0:
                # Non-autoregressive path: Apply prediction_length logic
                sequence_ids_for_preds_base = sequence_ids_tensor.numpy()
                item_positions_for_preds_base = (
                    start_positions_tensor.numpy() + config.seq_length
                )

                sequence_ids_for_preds = np.repeat(
                    sequence_ids_for_preds_base, prediction_length
                )

                item_positions_repeated = np.repeat(
                    item_positions_for_preds_base, prediction_length
                )
                position_offsets = np.tile(
                    np.arange(-prediction_length + 1, 1),
                    len(item_positions_for_preds_base),
                )
                item_positions_for_preds = item_positions_repeated + position_offsets

            else:
                sequence_ids_for_preds = np.repeat(
                    sequence_ids_tensor.numpy(), extra_steps + 1
                )
                item_position_boundaries = zip(
                    list(start_positions_tensor + config.seq_length),
                    list(start_positions_tensor + config.seq_length + extra_steps + 1),
                )
                item_positions_for_preds = np.concatenate(
                    [np.arange(start, end) for start, end in item_position_boundaries],
                    axis=0,
                )
            # --- END OF MODIFICATION ---
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

        os.makedirs(
            os.path.join(config.project_path, "outputs", "predictions"),
            exist_ok=True,
        )

        if config.output_probabilities:
            assert probs is not None
            os.makedirs(
                os.path.join(config.project_path, "outputs", "probabilities"),
                exist_ok=True,
            )

            for target_column in inferer.target_columns:
                if config.read_format in ["csv", "parquet"]:
                    file_name = f"{model_id}-{target_column}-probabilities.{config.write_format}"
                else:
                    dirname = f"{model_id}-{target_column}-probabilities"
                    file_name = os.path.join(
                        dirname,
                        f"{model_id}-{data_id}-probabilities.{config.write_format}",
                    )

                    dir_path = os.path.join(
                        config.project_path, "outputs", "probabilities", dirname
                    )
                    os.makedirs(dir_path, exist_ok=True)

                if inferer.target_column_types[target_column] == "categorical":
                    probabilities_path = os.path.join(
                        config.project_path, "outputs", "probabilities", file_name
                    )
                    print(f"[INFO] Writing probabilities to '{probabilities_path}'")
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

        if config.read_format in ["csv", "parquet"]:
            file_name = f"{model_id}-predictions.{config.write_format}"
        else:
            dirname = f"{model_id}-predictions"
            file_name = os.path.join(
                dirname, f"{model_id}-{data_id}-predictions.{config.write_format}"
            )
            dir_path = os.path.join(
                config.project_path, "outputs", "predictions", dirname
            )
            os.makedirs(dir_path, exist_ok=True)

        predictions_path = os.path.join(
            config.project_path, "outputs", "predictions", file_name
        )
        print(f"[INFO] Writing predictions to '{predictions_path}'")
        write_data(
            predictions,
            predictions_path,
            config.write_format,
        )


@beartype
def expand_data_by_autoregression(
    data: pl.DataFrame, autoregression_extra_steps: int, seq_length: int
) -> pl.DataFrame:
    """Expands a Polars DataFrame for autoregressive inference.

    This function takes a DataFrame of sequences and adds
    `autoregression_extra_steps` new rows for each sequence. These new
    rows represent future time steps to be predicted.

    For each new step, it:
    1. Copies the last known observation for a sequence.
    2. Increments the `subsequenceId`.
    3. Shifts the historical data columns (e.g., '1', '2', ..., '50') one
       position "older" (e.g., old '1' becomes new '2', old '49' becomes
       new '50').
    4. Fills the "newest" columns (e.g., new '1' for the first extra
       step) with `np.inf` as a placeholder for the prediction.

    Args:
        data: The input Polars DataFrame, sorted by `sequenceId` and
            `subsequenceId`.
        autoregression_extra_steps: The number of future time steps to add
            to each sequence.
        seq_length: The sequence length, used to identify the historical
            data columns (named '1' through `seq_length`).

    Returns:
        A new Polars DataFrame containing all original rows plus the
        newly generated future rows with placeholders.
    """
    # Ensure data is sorted for window functions
    data = data.sort("sequenceId", "subsequenceId")

    # Identify the last observation for each sequence
    last_obs_lazy = data.lazy().filter(
        pl.col("subsequenceId") == pl.col("subsequenceId").max().over("sequenceId")
    )

    # Generate future rows lazily
    future_frames = []
    data_cols = [str(c) for c in range(seq_length, 0, -1)]

    for offset in range(1, autoregression_extra_steps + 1):
        future_df_lazy = last_obs_lazy.with_columns(
            (pl.col("subsequenceId") + offset).alias("subsequenceId")
        ).with_columns(
            (pl.col("startItemPosition") + offset).alias("startItemPosition")
        )

        # Correctly shift columns to make space for future predictions
        # Newest value (col '1') becomes a placeholder, old '1' becomes new '2', etc.
        update_exprs = []
        for i in range(len(data_cols)):  # e.g., i from 0 to 49 for seq_length=50
            col_to_update = data_cols[i]  # Updates '50', '49', ..., '1'

            if i < len(data_cols) - offset:
                # Shift historical data one step further into the past
                # e.g., new '50' gets old '49', new '2' gets old '1'
                source_col_name = data_cols[i + offset]
                update_exprs.append(pl.col(source_col_name).alias(col_to_update))
            else:
                # These are the newest 'offset' columns, which are unknown.
                # Fill with infinity as a placeholder for the prediction.
                # e.g., for offset=1, new col '1' becomes inf.
                update_exprs.append(
                    pl.lit(np.inf, dtype=pl.Float64).alias(col_to_update)
                )

        future_df_lazy = future_df_lazy.with_columns(update_exprs)
        future_frames.append(future_df_lazy)

    # Concatenate original data with all future frames
    final_lazy = pl.concat([data.lazy()] + future_frames, how="vertical")

    return final_lazy.sort("sequenceId", "subsequenceId").collect()


@beartype
def get_embeddings_pt(
    config: Any,
    inferer: "Inferer",
    data: dict[str, torch.Tensor],
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
    X = {key: val.numpy() for key, val in data.items()}
    embeddings = inferer.infer_embedding(X)
    return embeddings


@beartype
def get_probs_preds_pt(
    config: Any,
    inferer: "Inferer",
    data: dict[str, torch.Tensor],
    extra_steps: int = 0,
) -> tuple[Optional[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Generates predictions from PyTorch tensor data, supporting autoregression.

    This function performs generative inference on a batch of PyTorch tensor
    data loaded from `.pt` files (which contain sequences, targets,
    sequence_ids, subsequence_ids, and start_positions). It implements an
    autoregressive loop:
    1. Runs inference on the initial data `X` (sequences).
    2. For each subsequent step (`i` in `extra_steps`):
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
        extra_steps: The number of additional autoregressive steps to
            perform. A value of 0 means simple, non-autoregressive
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
    all_probs_list = {col: [] for col in target_cols}
    all_preds_list = {col: [] for col in target_cols}

    # 3. Autoregressive loop
    # The loop runs `extra_steps + 1` times to get the initial prediction plus all extra steps.
    for i in range(extra_steps + 1):
        if config.output_probabilities:
            probs_for_step = inferer.infer_generative(X, return_probs=True)
            preds_for_step = inferer.infer_generative(None, probs_for_step)
            for col in target_cols:
                all_probs_list[col].append(probs_for_step[col])
        else:
            preds_for_step = inferer.infer_generative(X, return_probs=False)

        for col in target_cols:
            all_preds_list[col].append(preds_for_step[col])

        if i == extra_steps:
            break

        X_next = {}
        for col in X.keys():
            shifted_input = X[col][:, 1:]

            new_value = preds_for_step[col].reshape(-1, 1).astype(shifted_input.dtype)

            X_next[col] = np.concatenate([shifted_input, new_value], axis=1)

        X = X_next

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
    X = numpy_to_pytorch(data, column_types, all_columns, config.seq_length)
    X = {col: X_col.numpy() for col, X_col in X.items()}
    del data

    embeddings = inferer.infer_embedding(X)

    return embeddings


@beartype
def get_probs_preds(
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

    X = numpy_to_pytorch(data, column_types, all_columns, config.seq_length)
    X = {col: X_col.numpy() for col, X_col in X.items()}
    del data

    if config.output_probabilities:
        probs = inferer.infer_generative(X, return_probs=True)
        preds = inferer.infer_generative(None, probs)
    else:
        probs = None
        preds = inferer.infer_generative(X)

    return (probs, preds)


@beartype
def fill_in_predictions_pl(
    data: pl.DataFrame,
    preds: dict[str, np.ndarray],
    current_subsequence_id: int,
    sequence_ids_present: pl.Series,
    seq_length: int,
) -> pl.DataFrame:
    """Fills in predictions into the main Polars DataFrame using a robust,
    join-based approach that preserves the original DataFrame's structure.

    This function broadcasts predictions to all relevant future rows via a join,
    then uses conditional expressions to update only the specific placeholder
    cells (`np.inf`) that correspond to the correct future time step.

    Args:
        data: The main DataFrame containing all sequences.
        preds: A dictionary of new predictions, mapping target column names to NumPy arrays.
        current_subsequence_id: The adjusted subsequence ID at which predictions were made.
        sequence_ids_present: A Polars Series of the sequence IDs in the current batch.
        seq_length: The length of the sequence.

    Returns:
        An updated Polars DataFrame with the same dimensions as the input, with
        future placeholder values filled in.
    """
    if not preds or sequence_ids_present.is_empty():
        return data

    # 1. Create a "long" format DataFrame of the new predictions.
    # This table has columns [sequenceId, inputCol, prediction].
    pred_dfs = []
    for input_col, pred_values in preds.items():
        if pred_values.size > 0:
            pred_dfs.append(
                pl.DataFrame(
                    {
                        "sequenceId": sequence_ids_present,
                        "inputCol": input_col,
                        "prediction": pred_values.flatten(),
                    }
                )
            )

    # If there are no valid predictions to process, return the original data.
    if not pred_dfs:
        return data

    preds_df = pl.concat(pred_dfs)

    # 2. Left-join the predictions onto the main DataFrame.
    # This adds a 'prediction' column. Rows that don't match the join keys
    # (e.g., non-target columns) will have a null value for 'prediction'.
    # A left join guarantees that no rows from the original `data` are dropped.
    data_with_preds = data.join(preds_df, on=["sequenceId", "inputCol"], how="left")

    # 3. Build a list of conditional update expressions for each future time step.
    update_expressions = []
    for offset in range(1, seq_length + 1):
        col_to_update = str(offset)

        # Skip if the column to update doesn't exist in the DataFrame.
        if col_to_update not in data.columns:
            continue

        # The core logic: A prediction made at `current_subsequence_id` for a given
        # `offset` should fill the placeholder in column `str(offset)` at the row
        # where `subsequenceIdAdjusted` is `current_subsequence_id + offset`.
        update_expr = (
            pl.when(
                # Condition 1: Is this the correct future row to update?
                (pl.col("subsequenceIdAdjusted") == current_subsequence_id + offset)
                # Condition 2: Does the cell contain a placeholder that needs updating?
                & (pl.col(col_to_update).is_infinite())
                # Condition 3: Is there a valid prediction available from the join?
                & (pl.col("prediction").is_not_null())
            )
            # If all conditions are met, use the new prediction value.
            .then(pl.col("prediction"))
            # IMPORTANT: Otherwise, keep the column's existing value. This is crucial
            # for preserving the integrity of the DataFrame.
            .otherwise(pl.col(col_to_update))
            # Overwrite the original column with the updated values.
            .alias(col_to_update)
        )
        update_expressions.append(update_expr)

    # 4. Apply all expressions at once and remove the temporary 'prediction' column.
    # The `with_columns` operation does not change the number of rows.
    if update_expressions:
        updated_data = data_with_preds.with_columns(update_expressions).drop(
            "prediction"
        )
    else:
        # If no updates were needed, just drop the temporary join column.
        updated_data = data_with_preds.drop("prediction")

    return updated_data


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
    assert (
        is_globally_sorted
    ), "sequenceId must be in ascending order for autoregression"

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

    assert is_group_sorted, "subsequenceId must be in ascending order within each sequenceId for autoregression"


@beartype
def format_delta(time_delta: timedelta) -> str:
    """Formats a `timedelta` object into a human-readable string (seconds).

    Args:
        time_delta: The `timedelta` object to format.

    Returns:
        A string representing the total seconds with 3 decimal places.
    """
    seconds = time_delta.seconds
    microseconds = time_delta.microseconds
    return f"{(seconds + (microseconds/1e6)):.3}"


@beartype
def get_probs_preds_autoregression(
    config: Any,
    inferer: "Inferer",
    data: pl.DataFrame,
    column_types: dict[str, torch.dtype],
    seq_length: int,
) -> tuple[Optional[dict[str, np.ndarray]], dict[str, np.ndarray], np.ndarray]:
    """Performs autoregressive inference using a time-step-based Polars loop.

    This function orchestrates the autoregressive process by iterating
    through each unique, adjusted time step (`subsequenceIdAdjusted`).

    For each time step:
    1. Filters the main DataFrame `data` to get the current slice of data
       for all sequences at that time step.
    2. Calls `get_probs_preds` to generate predictions for this slice.
    3. Uses `fill_in_predictions_pl` to update the *main* `data` DataFrame,
       filling in the `np.inf` placeholders for the *next* time steps
       using the predictions just made.
    4. Collects the predictions and a corresponding sort key.

    After iterating through all time steps, it sorts all collected
    predictions based on the keys (sequenceId, subsequenceId) and returns
    the complete, ordered results.

    Args:
        config: The `InfererModel` configuration object.
        inferer: The initialized `Inferer` instance.
        data: The input Polars DataFrame, expanded with future rows
            (see `expand_data_by_autoregression`).
        column_types: A dictionary mapping column names to `torch.dtype`.
        seq_length: The sequence length, passed to `fill_in_predictions_pl`.

    Returns:
        A tuple `(probs, preds, sequence_ids)`:
            - `probs`: A dictionary mapping target columns to sorted NumPy
              arrays of probabilities, or `None`.
            - `preds`: A dictionary mapping target columns to sorted NumPy
              arrays of final predictions.
            - `sequence_ids`: A NumPy array of `sequenceId`s corresponding
              to each row in the `preds` arrays.
    """
    data = data.sort("sequenceId", "subsequenceId")
    verify_variable_order(data)

    # Normalize subsequenceId to start from 0 for each sequence
    data = data.with_columns(
        (
            pl.col("subsequenceId") - pl.col("subsequenceId").min().over("sequenceId")
        ).alias("subsequenceIdAdjusted")
    )

    preds_list, probs_list, sort_keys = [], [], []
    subsequence_ids_distinct = sorted(data["subsequenceIdAdjusted"].unique().to_list())

    # Ensure max_length for padding is robust for both sequence and subsequence IDs
    max_id_val = max(
        data["sequenceId"].max(),
        max(subsequence_ids_distinct) if subsequence_ids_distinct else 0,
    )
    max_length = len(str(max_id_val))

    for subsequence_id in subsequence_ids_distinct:
        t0 = datetime.now()
        data_subset = data.filter(pl.col("subsequenceIdAdjusted") == subsequence_id)

        if data_subset.height == 0:
            continue

        sequence_ids_present = data_subset.get_column("sequenceId").unique(
            maintain_order=True
        )

        t1 = datetime.now()

        # Original sort key logic
        sort_keys.extend(
            [
                f"{fill_number(int(seq_id), max_length)}-{fill_number(int(subsequence_id), max_length)}"
                for seq_id in sequence_ids_present
            ]
        )

        t2 = datetime.now()

        probs, preds = get_probs_preds(
            config,
            inferer,
            data_subset,
            column_types,
        )

        t3 = datetime.now()

        preds_list.append(preds)
        if probs is not None:
            probs_list.append(probs)

        # Use new Polars-native function to fill predictions
        data = fill_in_predictions_pl(
            data,
            preds,
            int(subsequence_id),
            sequence_ids_present,
            seq_length,
        )

        t4 = datetime.now()

        print(
            f"[DEBUG] Autoregression step {subsequence_id}/{len(subsequence_ids_distinct)}: Total: {format_delta(t4-t0)}s (Filter: {format_delta(t1-t0)}s, Infer: {format_delta(t3-t2)}s, Update: {format_delta(t4-t3)}s)"
        )

    sort_order = np.argsort(sort_keys)

    preds = {
        target_column: np.concatenate([p[target_column] for p in preds_list], axis=0)[
            sort_order
        ]
        for target_column in inferer.target_columns
    }
    if len(probs_list):
        probs = {
            target_column: np.concatenate(
                [p[target_column] for p in probs_list], axis=0
            )[sort_order, :]
            for target_column in inferer.target_columns
        }
    else:
        probs = None

    # Create the corresponding sequence_id array from the sorted keys
    sorted_keys = np.array(sort_keys)[sort_order]
    sequence_ids_for_preds = np.array([int(key.split("-")[0]) for key in sorted_keys])

    return probs, preds, sequence_ids_for_preds


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
        project_path: str,
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
            project_path: The path to the sequifier project directory.
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
                normalize_path(model_path, project_path),
                providers=execution_providers,
                **kwargs,
            )
        if self.inference_model_type == "pt":
            self.inference_model = load_inference_model(
                self.model_type,
                normalize_path(model_path, project_path),
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
        embedding = self.adjust_and_infer_embedding(x, size)

        return embedding

    @beartype
    def infer_generative(
        self,
        x: Optional[dict[str, np.ndarray]],
        probs: Optional[dict[str, np.ndarray]] = None,
        return_probs: bool = False,
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

            outs = self.adjust_and_infer_generative(x, size)

            for target_column, target_outs in outs.items():
                assert not np.any(target_outs == np.inf), target_outs

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
                    outs[target_column] = sample_with_cumsum(outs[target_column])

        return outs

    @beartype
    def adjust_and_infer_embedding(self, x: dict[str, np.ndarray], size: int):
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
            inference_batch_embeddings = [
                self.infer_pure(x_sub)[0][:size] for x_sub in x_adjusted
            ]
            embeddings = np.concatenate(inference_batch_embeddings, axis=0)
        elif self.inference_model_type == "pt":
            x_adjusted = self.prepare_inference_batches(x, pad_to_batch_size=False)
            embeddings = infer_with_embedding_model(
                self.inference_model,
                x_adjusted,
                self.device,
                size,
                self.target_columns,
            )
        else:
            assert False, "not possible"
        return embeddings

    @beartype
    def adjust_and_infer_generative(self, x: dict[str, np.ndarray], size: int):
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
            out_subs = [
                dict(zip(self.target_columns, self.infer_pure(x_sub)))
                for x_sub in x_adjusted
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
            outs = infer_with_generative_model(
                self.inference_model,
                x_adjusted,
                self.device,
                size * self.prediction_length,
                self.target_columns,
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
    def infer_pure(self, x: dict[str, np.ndarray]) -> list[np.ndarray]:
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
        ort_inputs = {
            session_input.name: self.expand_to_batch_size(x[col])
            for session_input, col in zip(
                self.ort_session.get_inputs(),
                self.categorical_columns + self.real_columns,
            )
        }
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
def sample_with_cumsum(probs: np.ndarray) -> np.ndarray:
    """Samples from a probability distribution using the inverse CDF method.

    Takes an array of logits, computes the cumulative probability
    distribution, draws a random number `r` from [0, 1), and returns
    the index of the first class `i` where `cumsum[i] > r`.

    Args:
        probs: A 2D NumPy array of *logits* (not normalized probabilities).
               Shape is (batch_size, num_classes).

    Returns:
        A 1D NumPy array of shape (batch_size,) containing the sampled
        class indices.
    """
    cumulative_probs = np.cumsum(np.exp(probs), axis=1)
    random_threshold = np.random.rand(cumulative_probs.shape[0], 1)
    random_threshold = np.repeat(random_threshold, probs.shape[1], axis=1)
    return (random_threshold < cumulative_probs).argmax(axis=1)
