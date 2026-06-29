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
    normalize_path,
    numpy_to_pytorch,
    resolve_unified_polars_numeric_dtype,
    resolve_window_view,
    subset_to_input_columns,
    validate_stored_window_width,
    write_data,
)
from sequifier.special_tokens import validate_special_token_ids
from sequifier.train import (
    infer_with_embedding_model,
    infer_with_generative_model,
    load_inference_model,
)

ONNX_NUMPY_DTYPES = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
    "tensor(bool)": np.bool_,
}


@beartype
def infer(args: Any, args_config: dict[str, Any]) -> None:
    """Load inference config and dispatch the worker."""
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
                metadata_config["special_token_ids"],
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
    """Yield a percentage slice of sorted top-level PT files."""
    pt_files = sorted(Path(data_path).glob("*.pt"))

    total = len(pt_files)
    start_idx = int(total * start_pct / 100)
    end_idx = int(total * end_pct / 100)

    for pt_file in pt_files[start_idx:end_idx]:
        yield torch.load(pt_file, weights_only=False)


@beartype
def load_parquet_folder_dataset(
    data_path: str, start_pct: float, end_pct: float
) -> Iterator[Any]:
    """Yield a percentage slice of sorted top-level Parquet files."""
    parquet_files = sorted(Path(data_path).glob("*.parquet"))

    total = len(parquet_files)
    start_idx = int(total * start_pct / 100)
    end_idx = int(total * end_pct / 100)

    for parquet_file in parquet_files[start_idx:end_idx]:
        yield pl.read_parquet(parquet_file)


@beartype
def _torch_column_types(config: InfererModel) -> dict[str, torch.dtype]:
    return {
        col: PANDAS_TO_TORCH_TYPES[config.column_types[col]]
        for col in config.column_types
    }


@beartype
def _sequence_position_columns(config: InfererModel, data: pl.DataFrame) -> list[str]:
    return [
        str(i)
        for i in range(config.storage_layout.stored_context_width - 1, -1, -1)
        if str(i) in data.columns
    ]


@beartype
def _configured_types_for_loaded_rows(
    config: InfererModel, data: pl.DataFrame
) -> dict[str, str]:
    if "inputCol" not in data.columns:
        return {
            column: config.column_types[column]
            for column in config.input_columns
            if column in config.column_types
        }

    loaded_columns = [
        column
        for column in data.get_column("inputCol").unique().to_list()
        if column in config.column_types
    ]
    return {column: config.column_types[column] for column in loaded_columns}


@beartype
def apply_inference_column_types(
    data: pl.DataFrame, config: InfererModel
) -> pl.DataFrame:
    """Cast loaded long-format sequence values to the configured unified dtype."""
    sequence_columns = _sequence_position_columns(config, data)
    if not sequence_columns:
        return data

    configured_types = _configured_types_for_loaded_rows(config, data)
    if not configured_types:
        return data

    unified_dtype = resolve_unified_polars_numeric_dtype(configured_types)
    casts = [
        pl.col(column).cast(unified_dtype)
        for column in sequence_columns
        if data.schema[column] != unified_dtype
    ]
    if not casts:
        return data
    return data.with_columns(casts)


@beartype
def apply_inference_tensor_types(
    sequences_dict: dict[str, torch.Tensor],
    column_types: dict[str, torch.dtype],
) -> dict[str, torch.Tensor]:
    """Cast loaded PT feature tensors to the configured per-column dtype."""
    return {
        column: tensor.to(dtype=column_types[column])
        if column in column_types and tensor.dtype != column_types[column]
        else tensor
        for column, tensor in sequences_dict.items()
    }


@beartype
def infer_worker(
    config: Any,
    args_config: dict[str, Any],
    id_maps: Optional[dict[str, dict[str | int, int]]],
    selected_columns_statistics: dict[str, dict[str, float]],
    percentage_limits: Optional[tuple[float, float]],
):
    """Load data, instantiate models, and run the configured inference mode."""
    logger.info(f"[INFO] Reading data from '{config.data_path}'...")

    is_folder_input = os.path.isdir(
        normalize_path(config.data_path, config.project_root)
    )
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

        default_prediction_length = {
            "causal": 1,
            "final_value": 1,
            "next_occurrence": 1,
            "bert": config.window_view.context_length,
        }
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

        column_types = _torch_column_types(config)

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
    context_length: int,
    prediction_length: int,
    training_objective: str,
) -> np.ndarray:
    """Return flattened absolute item positions for inference outputs."""
    if training_objective == "bert":
        # Anchor positions to the start of the input sequence and tile forwards
        base_positions = start_positions
        position_offsets = np.arange(0, prediction_length)
    elif training_objective in ["causal", "final_value", "next_occurrence"]:
        # Anchor positions to the future token step and tile backwards
        base_positions = start_positions + context_length
        position_offsets = np.arange(-prediction_length + 1, 1)
    else:
        raise ValueError(f"Unknown objective {training_objective}")

    # Repeat base anchors to match the number of predictions per sequence window
    repeated_bases = np.repeat(base_positions, prediction_length)
    # Tile the relative step offsets across all sequences in the batch chunk
    tiled_offsets = np.tile(position_offsets, len(start_positions))

    return repeated_bases + tiled_offsets


@beartype
def _flatten_valid_mask(
    config: InfererModel,
    metadata: dict[str, Any],
    prediction_length: int,
    mask_key: str = "target_valid_mask",
) -> np.ndarray:
    valid_mask = metadata.get(mask_key)
    if isinstance(valid_mask, torch.Tensor):
        valid_mask = valid_mask.detach().cpu().numpy()  # type: ignore
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if valid_mask.ndim != 2:
        raise ValueError(f"{mask_key} must be 2D, got shape {valid_mask.shape}.")

    return valid_mask[:, -prediction_length:].reshape(-1)


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
def _valid_mask_from_preprocessed_data(
    config: InfererModel,
    data: pl.DataFrame,
    prediction_length: int,
    mask_key: str = "target_valid_mask",
) -> np.ndarray:
    data_columns = set(data.get_column("inputCol").unique())
    column_name = _bert_reference_column(config, data_columns)
    reference_rows = data.filter(pl.col("inputCol") == column_name)

    left_pad_lengths = torch.tensor(
        reference_rows.get_column("leftPadLength").to_numpy(), dtype=torch.int64
    )
    resolved_view = resolve_window_view(config.storage_layout, config.window_view)
    metadata = resolved_view.build_masks(left_pad_lengths)
    return _flatten_valid_mask(config, metadata, prediction_length, mask_key)


@beartype
def _apply_valid_prediction_mask(
    values: np.ndarray,
    valid_prediction_mask: np.ndarray,
    label: str,
) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[0] != valid_prediction_mask.shape[0]:
        raise ValueError(
            f"{label} has {values.shape[0]} rows, but valid_prediction_mask has "
            f"{valid_prediction_mask.shape[0]} rows."
        )
    return values[valid_prediction_mask]


@beartype
def _apply_valid_prediction_mask_to_dict(
    values: Optional[dict[str, np.ndarray]],
    valid_prediction_mask: np.ndarray,
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
    """Write embeddings for each dataset chunk."""
    for data_id, data in enumerate(dataset):
        prediction_length = inferer.prediction_length
        valid_prediction_mask = None

        is_folder_input = os.path.isdir(
            normalize_path(config.data_path, config.project_root)
        )

        if config.read_format in ["parquet", "csv"] and not is_folder_input:
            if config.input_columns is not None:
                data = subset_to_input_columns(data, config.input_columns)
            data = apply_inference_column_types(data, config)

            # Determine the number of input features
            n_input_cols = data.get_column("inputCol").n_unique()

            # Create a mask to select only one row per sequence
            mask = pl.arange(0, data.height, eager=True) % n_input_cols == 0

            embeddings = get_embeddings(config, inferer, data, column_types)
            valid_prediction_mask = _valid_mask_from_preprocessed_data(
                config, data, prediction_length, mask_key="attention_valid_mask"
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
            data = apply_inference_column_types(data, config)

            n_input_cols = data.get_column("inputCol").n_unique()
            mask = pl.arange(0, data.height, eager=True) % n_input_cols == 0

            embeddings = get_embeddings(config, inferer, data, column_types)
            valid_prediction_mask = _valid_mask_from_preprocessed_data(
                config, data, prediction_length, mask_key="attention_valid_mask"
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
            ) = data
            sequences_dict = apply_inference_tensor_types(sequences_dict, column_types)
            for tensor in sequences_dict.values():
                validate_stored_window_width(
                    tensor, config.storage_layout.stored_context_width
                )

            resolved_view = resolve_window_view(
                config.storage_layout, config.window_view
            )
            metadata = resolved_view.build_masks(left_pad_lengths_tensor)
            embeddings = get_embeddings_pt(
                config,
                inferer,
                sequences_dict,
                metadata=metadata,
                column_types=column_types,
            )
            valid_prediction_mask = _flatten_valid_mask(
                config, metadata, prediction_length, mask_key="attention_valid_mask"
            )

            sequence_ids_for_preds = sequence_ids_tensor.numpy()
            subsequence_ids_for_preds = subsequence_ids_tensor.numpy()
            item_positions_for_preds_base = start_positions_tensor.numpy()

        else:
            raise Exception("impossible")

        # Step 2: Calculate absolute positions and repeat IDs
        # (e.g., for seq_len=50, inf_size=5, offsets are [45, 46, 47, 48, 49])
        base_offsets = np.arange(
            config.window_view.context_length - prediction_length,
            config.window_view.context_length,
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
        assert valid_prediction_mask is not None
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
    """Write generative predictions/probabilities for each dataset chunk."""
    for data_id, data in enumerate(dataset):
        is_folder_input = os.path.isdir(
            normalize_path(config.data_path, config.project_root)
        )

        if config.read_format in ["parquet", "csv"] and not is_folder_input:
            if config.input_columns is not None:
                data = subset_to_input_columns(data, config.input_columns)
            data = apply_inference_column_types(data, config)
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
                valid_prediction_mask = _valid_mask_from_preprocessed_data(
                    config, data, prediction_length
                )

                # Expand IDs to match model output shape
                sequence_ids_for_preds = np.repeat(
                    sequence_ids_for_preds_base, prediction_length
                )

                # Invoke the unified positioning engine
                item_positions_for_preds = calculate_item_positions(
                    item_positions_base_raw,
                    config.window_view.context_length,
                    prediction_length,
                    config.training_objective,
                )

            else:
                if inferer.prediction_length != 1:
                    raise ValueError(
                        f"prediction_length must be 1 for autoregression, got {inferer.prediction_length}"
                    )
                # Unpack the new third return value
                (
                    probs,
                    preds,
                    sequence_ids_for_preds,
                    item_positions_for_preds,
                    valid_prediction_mask,
                ) = get_probs_preds_autoregression(
                    config,
                    inferer,
                    data,
                    column_types,
                    config.window_view.context_length,
                )
        elif config.read_format == "parquet" and is_folder_input:
            # Folder-based Parquet chunk logic
            if config.input_columns is not None:
                data = subset_to_input_columns(data, config.input_columns)
            data = apply_inference_column_types(data, config)
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
                valid_prediction_mask = _valid_mask_from_preprocessed_data(
                    config, data, prediction_length
                )

                sequence_ids_for_preds = np.repeat(
                    sequence_ids_for_preds_base, prediction_length
                )

                # Invoke the unified positioning engine
                item_positions_for_preds = calculate_item_positions(
                    item_positions_base_raw,
                    config.window_view.context_length,
                    prediction_length,
                    config.training_objective,
                )
            else:
                if inferer.prediction_length != 1:
                    raise ValueError(
                        f"prediction_length must be 1 for autoregression, got {inferer.prediction_length}"
                    )

                (
                    probs,
                    preds,
                    sequence_ids_for_preds,
                    item_positions_for_preds,
                    valid_prediction_mask,
                ) = get_probs_preds_autoregression(
                    config,
                    inferer,
                    data,
                    column_types,
                    config.window_view.context_length,
                )
        elif config.read_format == "pt":
            (
                sequences_dict,
                sequence_ids_tensor,
                _,
                start_positions_tensor,
                left_pad_lengths_tensor,
            ) = data
            sequences_dict = apply_inference_tensor_types(sequences_dict, column_types)
            for tensor in sequences_dict.values():
                validate_stored_window_width(
                    tensor, config.storage_layout.stored_context_width
                )
            total_steps = (
                1
                if config.autoregression_total_steps is None
                else config.autoregression_total_steps
            )

            resolved_view = resolve_window_view(
                config.storage_layout, config.window_view
            )
            sequences_dict = {
                key: tensor[:, resolved_view.input_slice]
                for key, tensor in sequences_dict.items()
                if key in config.input_columns
            }

            metadata = resolved_view.build_masks(left_pad_lengths_tensor)

            probs, preds = get_probs_preds_from_dict(
                config,
                inferer,
                sequences_dict,
                metadata,
                column_types,
                total_steps,
            )

            prediction_length = inferer.prediction_length  # Get prediction_length
            valid_prediction_mask = _flatten_valid_mask(
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
                    config.window_view.context_length,
                    prediction_length,
                    config.training_objective,
                )

            else:
                sequence_ids_for_preds = np.repeat(
                    sequence_ids_tensor.numpy(), total_steps
                )
                valid_prediction_mask = np.repeat(valid_prediction_mask, total_steps)
                item_position_boundaries = zip(
                    list(start_positions_tensor + config.window_view.context_length),
                    list(
                        start_positions_tensor
                        + config.window_view.context_length
                        + total_steps
                    ),
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

        assert valid_prediction_mask is not None

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
    metadata: dict[str, torch.Tensor],
    column_types: dict[str, torch.dtype],
) -> np.ndarray:
    """Infer embeddings from PT tensors."""
    resolved_view = resolve_window_view(config.storage_layout, config.window_view)
    for tensor in data.values():
        validate_stored_window_width(tensor, config.storage_layout.stored_context_width)
    X = {
        key: val[:, resolved_view.input_slice].numpy()
        for key, val in data.items()
        if key in config.input_columns
    }
    metadata_np = {key: val.numpy() for key, val in metadata.items()}
    embeddings = inferer.infer_embedding(
        X, metadata=metadata_np, column_types=column_types
    )
    return embeddings


@beartype
def get_probs_preds_from_dict(
    config: Any,
    inferer: "Inferer",
    data: dict[str, torch.Tensor],
    metadata: dict[str, torch.Tensor],
    column_types: dict[str, torch.dtype],
    total_steps: int = 1,
) -> tuple[Optional[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Infer PT predictions, flattened sample-major across autoregressive steps."""

    target_cols = inferer.target_columns

    X = {
        key: tensor.numpy()
        for key, tensor in data.items()
        if key in config.input_columns
    }
    metadata_np = {key: tensor.numpy() for key, tensor in metadata.items()}
    all_probs_list = {col: [] for col in target_cols}
    all_preds_list = {col: [] for col in target_cols}

    metadata_for_step = metadata_np
    for i in range(total_steps):
        if config.output_probabilities:
            probs_for_step = inferer.infer_generative(
                X,
                metadata_for_step,
                column_types=column_types,
                return_probs=True,
            )
            preds_for_step = inferer.infer_generative(
                None, metadata_for_step, probs_for_step
            )
            for col in target_cols:
                all_probs_list[col].append(probs_for_step[col])
        else:
            preds_for_step = inferer.infer_generative(
                X,
                metadata_for_step,
                column_types=column_types,
                return_probs=False,
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
    """Infer embeddings from a Polars chunk."""
    all_columns = sorted(list(set(config.input_columns + config.target_columns)))
    resolved_view = resolve_window_view(config.storage_layout, config.window_view)
    X, metadata = numpy_to_pytorch(
        data,
        column_types,
        all_columns,
        resolved_view,
    )
    X = {col: X_col.numpy() for col, X_col in X.items()}
    metadata_np = {col: metadata_col.numpy() for col, metadata_col in metadata.items()}
    del data

    embeddings = inferer.infer_embedding(
        X, metadata=metadata_np, column_types=column_types
    )

    return embeddings


@beartype
def get_probs_preds_from_df(
    config: Any,
    inferer: "Inferer",
    data: pl.DataFrame,
    column_types: dict[str, torch.dtype],
) -> tuple[Optional[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Infer non-autoregressive predictions from a Polars chunk."""
    all_columns = sorted(list(set(config.input_columns + config.target_columns)))

    resolved_view = resolve_window_view(config.storage_layout, config.window_view)
    X, metadata = numpy_to_pytorch(
        data,
        column_types,
        all_columns,
        resolved_view,
    )
    X = {col: X_col.numpy() for col, X_col in X.items()}
    metadata_np = {col: metadata_col.numpy() for col, metadata_col in metadata.items()}
    del data

    if config.output_probabilities:
        probs = inferer.infer_generative(
            X, metadata_np, column_types=column_types, return_probs=True
        )
        preds = inferer.infer_generative(None, metadata_np, probs)
    else:
        probs = None
        preds = inferer.infer_generative(X, metadata_np, column_types=column_types)

    return (probs, preds)


@beartype
def fill_number(number: Union[int, float], max_length: int) -> str:
    """Left-pad a number for sortable string keys."""
    number_str = str(number)
    return f"{'0' * (max_length - len(number_str))}{number_str}"


@beartype
def verify_variable_order(data: pl.DataFrame) -> None:
    """Require sequenceId order and in-sequence subsequenceId order."""
    is_globally_sorted = data.select(
        (pl.col("sequenceId").diff().fill_null(0) >= 0).all()
    ).item()
    if not is_globally_sorted:
        raise ValueError("sequenceId must be in ascending order for autoregression")

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
    context_length: int,
) -> tuple[
    Optional[dict[str, np.ndarray]],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Infer autoregressive predictions with sequence IDs, positions, and mask."""
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
        + context_length
    )

    resolved_view = resolve_window_view(config.storage_layout, config.window_view)
    head_data, metadata = numpy_to_pytorch(
        head_data_df,
        column_types,
        config.input_columns,
        resolved_view,
    )

    probs, preds = get_probs_preds_from_dict(
        config,
        inferer,
        head_data,
        metadata,
        column_types,
        config.autoregression_total_steps,
    )

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

    base_mask = _flatten_valid_mask(config, metadata, 1)
    valid_prediction_mask = np.repeat(base_mask, config.autoregression_total_steps)

    return (
        probs,
        preds,
        sequence_ids_for_preds,
        item_positions_for_preds,
        valid_prediction_mask,
    )


class Inferer:
    """Inference runtime for PT/ONNX sequifier models."""

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
        """Load a PT or ONNX backend and postprocessing state."""
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
        """Invert target-column Z-score normalization."""
        std = self.selected_columns_statistics[target_column]["std"]
        mean = self.selected_columns_statistics[target_column]["mean"]
        return (values * (std - 1e-9)) + mean

    @beartype
    def infer_embedding(
        self,
        x: dict[str, np.ndarray],
        metadata: dict[str, np.ndarray],
        column_types: dict[str, torch.dtype],
    ) -> np.ndarray:
        """Return embeddings for a feature-array batch."""
        assert x is not None
        size = x[list(x.keys())[0]].shape[0]
        embedding = self.adjust_and_infer_embedding(x, size, metadata, column_types)

        return embedding

    @beartype
    def infer_generative(
        self,
        x: Optional[dict[str, np.ndarray]],
        metadata: dict[str, np.ndarray],
        probs: Optional[dict[str, np.ndarray]] = None,
        return_probs: bool = False,
        column_types: Optional[dict[str, torch.dtype]] = None,
    ) -> dict[str, np.ndarray]:
        """Return target probabilities or decoded predictions."""
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

            outs = self.adjust_and_infer_generative(
                x, size, metadata, column_types or {}
            )

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
        metadata: dict[str, np.ndarray],
        column_types: dict[str, torch.dtype],
    ):
        """Batch embedding inference across the active backend."""
        if self.inference_model_type == "onnx":
            assert x is not None
            x_adjusted = self.prepare_inference_batches(x, pad_to_batch_size=True)
            metadata_adjusted = self.prepare_inference_batches(
                metadata, pad_to_batch_size=True
            )

            inference_batch_embeddings = [
                self.infer_pure(x_sub, metadata_sub)[0]
                for x_sub, metadata_sub in zip(x_adjusted, metadata_adjusted)
            ]
            embeddings = np.concatenate(inference_batch_embeddings, axis=0)[
                : size * self.prediction_length
            ]
        elif self.inference_model_type == "pt":
            x_adjusted = self.prepare_inference_batches(x, pad_to_batch_size=False)
            metadata_adjusted = self.prepare_inference_batches(
                metadata, pad_to_batch_size=False
            )

            embeddings = infer_with_embedding_model(
                self.inference_model,
                x_adjusted,
                self.device,
                size,
                self.target_columns,
                metadata=metadata_adjusted,
                column_types=column_types,
            )
        else:
            assert False, "not possible"
        return embeddings

    @beartype
    def adjust_and_infer_generative(
        self,
        x: dict[str, np.ndarray],
        size: int,
        metadata: dict[str, np.ndarray],
        column_types: dict[str, torch.dtype],
    ):
        """Batch generative inference across the active backend."""
        if self.inference_model_type == "onnx":
            assert x is not None
            x_adjusted = self.prepare_inference_batches(x, pad_to_batch_size=True)
            metadata_adjusted = self.prepare_inference_batches(
                metadata, pad_to_batch_size=True
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
            metadata_adjusted = self.prepare_inference_batches(
                metadata, pad_to_batch_size=False
            )
            outs = infer_with_generative_model(
                self.inference_model,
                x_adjusted,
                self.device,
                size * self.prediction_length,
                self.target_columns,
                metadata=metadata_adjusted,
                column_types=column_types,
            )
        else:
            assert False
            outs = {}  # for type checking

        return outs

    @beartype
    def prepare_inference_batches(
        self, x: dict[str, np.ndarray], pad_to_batch_size: bool
    ) -> list[dict[str, np.ndarray]]:
        """Split feature arrays into backend-sized batches."""
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
        metadata: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        """Run one ONNX batch and flatten sequence-major outputs."""
        metadata = metadata or {}
        ort_inputs = {}
        for session_input in self.ort_session.get_inputs():
            input_name = session_input.name
            if input_name in metadata:
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

            expected_dtype = ONNX_NUMPY_DTYPES.get(session_input.type)
            if expected_dtype is not None and value.dtype != expected_dtype:
                value = value.astype(expected_dtype, copy=False)
            ort_inputs[input_name] = self.expand_to_batch_size(value)

        ort_outs = self.ort_session.run(None, ort_inputs)
        return [
            oo.transpose(1, 0, 2).reshape(oo.shape[0] * oo.shape[1], oo.shape[2])
            for oo in ort_outs
        ]

    @beartype
    def expand_to_batch_size(self, x: np.ndarray) -> np.ndarray:
        """Repeat leading samples until the ONNX batch size is met."""
        repetitions = self.inference_batch_size // x.shape[0]
        filler = self.inference_batch_size % x.shape[0]
        return np.concatenate(([x] * repetitions) + [x[0:filler, :]], axis=0)


@beartype
def normalize(outs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Softmax logits by target column."""
    shifted_values = {
        target_column: target_values - np.max(target_values, axis=1, keepdims=True)
        for target_column, target_values in outs.items()
    }
    exp_values = {
        target_column: np.exp(target_values)
        for target_column, target_values in shifted_values.items()
    }
    probs = {
        target_column: target_values / np.sum(target_values, axis=1, keepdims=True)
        for target_column, target_values in exp_values.items()
    }
    return probs


@beartype
def sample_with_cumsum(probs: np.ndarray, is_log_probs: bool = True) -> np.ndarray:
    """Sample class indices from log-probabilities or probabilities."""
    if is_log_probs:
        sampling_probs = np.exp(probs)
    else:
        sampling_probs = probs

    if not np.isfinite(sampling_probs).all():
        raise ValueError("Sampling probabilities must be finite.")
    if np.any(sampling_probs < 0):
        raise ValueError("Sampling probabilities must be non-negative.")

    row_sums = sampling_probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError("Sampling probabilities must sum to 1.0 for each row.")

    cumulative_probs = np.cumsum(sampling_probs, axis=1)
    cumulative_probs[:, -1] = 1.0
    random_threshold = np.random.rand(cumulative_probs.shape[0], 1)
    random_threshold = np.repeat(random_threshold, probs.shape[1], axis=1)
    return (random_threshold < cumulative_probs).argmax(axis=1)
