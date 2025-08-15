import json
import os
import warnings
from datetime import datetime
from typing import Any, Optional, Union

import numpy as np
import onnxruntime
import polars as pl
import torch
from beartype import beartype

from sequifier.config.infer_config import load_inferer_config
from sequifier.helpers import (
    PANDAS_TO_TORCH_TYPES,
    construct_index_maps,
    normalize_path,
    numpy_to_pytorch,
    subset_to_selected_columns,
    write_data,
)
from sequifier.train import infer_with_model, load_inference_model


@beartype
def infer(args: Any, args_config: dict[str, Any]) -> None:
    print("Inferring...")
    config_path = (
        args.config_path if args.config_path is not None else "configs/infer.yaml"
    )

    config = load_inferer_config(config_path, args_config, args.on_unprocessed)

    if config.map_to_id or (len(config.real_columns) > 0):
        assert config.ddconfig_path is not None, (
            "If you want to map to id, you need to provide a file path to a json that contains: {{'id_maps':{...}}} to ddconfig_path"
            "\nIf you have real columns in the data, you need to provide a json that contains: {{'selected_columns_statistics':{COL_NAME:{'std':..., 'mean':...}}}}"
        )
        with open(normalize_path(config.ddconfig_path, config.project_path), "r") as f:
            dd_config = json.loads(f.read())
            id_maps = dd_config["id_maps"]
            selected_columns_statistics = dd_config["selected_columns_statistics"]
    else:
        id_maps = None
        selected_columns_statistics = {}

    print("Reading data...")
    # Step 1: Use Polars for data ingestion
    if config.read_format == "parquet":
        data = pl.read_parquet(config.data_path)
    else:
        data = pl.read_csv(config.data_path)

    model_paths = (
        config.model_path
        if isinstance(config.model_path, list)
        else [config.model_path]
    )
    for model_path in model_paths:
        inferer = Inferer(
            model_path,
            config.project_path,
            id_maps,
            selected_columns_statistics,
            config.map_to_id,
            config.categorical_columns,
            config.real_columns,
            config.selected_columns,
            config.target_columns,
            config.target_column_types,
            config.sample_from_distribution_columns,
            config.infer_with_dropout,
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

        print(f"Inferring for {model_id}")

        # Step 1: Adapt Data Subsetting (now works on Polars DF)
        if config.selected_columns is not None:
            data = subset_to_selected_columns(data, config.selected_columns)
        if not config.autoregression:
            # For the non-autoregressive case, the old logic is still needed here
            n_input_cols = data.get_column("inputCol").n_unique()
            mask = pl.arange(0, data.height, eager=True) % n_input_cols == 0
            # Apply the mask to the sequenceId column
            sequence_ids_for_preds = data.get_column("sequenceId").filter(mask)
            probs, preds = get_probs_preds(
                config,
                inferer,
                data,
                column_types,
                apply_normalization_inversion=False,
            )
        else:
            if config.autoregression_additional_steps is not None:
                data = expand_data_by_autoregression(
                    data, config.autoregression_additional_steps, config.seq_length
                )

            # Unpack the new third return value
            probs, preds, sequence_ids_for_preds = get_probs_preds_autoregression(
                config, inferer, data, column_types, config.seq_length
            )

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
            os.path.join(config.project_path, "outputs", "predictions"), exist_ok=True
        )

        if config.output_probabilities:
            assert probs is not None
            os.makedirs(
                os.path.join(config.project_path, "outputs", "probabilities"),
                exist_ok=True,
            )
            for target_column in inferer.target_columns:
                if inferer.target_column_types[target_column] == "categorical":
                    probabilities_path = os.path.join(
                        config.project_path,
                        "outputs",
                        "probabilities",
                        f"{model_id}-{target_column}-probabilities.{config.write_format}",
                    )
                    print(f"Writing probabilities to {probabilities_path}")
                    # Step 5: Finalize Output and I/O (write_data now handles Polars DF)
                    write_data(
                        pl.DataFrame(
                            probs[target_column],
                            schema=[
                                str(i) for i in range(probs[target_column].shape[1])
                            ],
                        ),
                        probabilities_path,
                        config.write_format,
                    )

        n_input_cols = data.get_column("inputCol").n_unique()
        predictions = pl.DataFrame(
            {
                "sequenceId": sequence_ids_for_preds,
                **{
                    target_column: preds[target_column].flatten()
                    for target_column in inferer.target_columns
                },
            }
        )

        predictions_path = os.path.join(
            config.project_path,
            "outputs",
            "predictions",
            f"{model_id}-predictions.{config.write_format}",
        )
        print(f"Writing predictions to {predictions_path}")
        write_data(
            predictions,
            predictions_path,
            config.write_format,
        )
        print("Inference complete")


@beartype
def expand_data_by_autoregression(
    data: pl.DataFrame, autoregression_additional_steps: int, seq_length: int
) -> pl.DataFrame:
    """
    Expand data for autoregression by adding additional steps using Polars.
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

    for offset in range(1, autoregression_additional_steps + 1):
        future_df_lazy = last_obs_lazy.with_columns(
            (pl.col("subsequenceId") + offset).alias("subsequenceId")
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
def get_probs_preds(
    config: Any,
    inferer: "Inferer",
    data: pl.DataFrame,
    column_types: dict[str, torch.dtype],
    apply_normalization_inversion: bool = True,
) -> tuple[Optional[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    X, _ = numpy_to_pytorch(
        data,
        column_types,
        config.selected_columns,
        config.target_columns,
        config.seq_length,
        config.device,
        to_device=False,
    )
    X = {col: X_col.numpy() for col, X_col in X.items()}
    del data

    if config.output_probabilities:
        probs = inferer.infer(X, return_probs=True)
        preds = inferer.infer(
            None, probs, apply_normalization_inversion=apply_normalization_inversion
        )
    else:
        probs = None

        preds = inferer.infer(
            X, apply_normalization_inversion=apply_normalization_inversion
        )

    return (probs, preds)


@beartype
def fill_in_predictions_pl(
    data: pl.DataFrame,
    preds: dict[str, np.ndarray],
    current_subsequence_id: int,
    sequence_ids_present: pl.Series,
    seq_length: int,
) -> pl.DataFrame:
    """
    Fills in predictions into the main Polars DataFrame using a robust,
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
    """
    Fill a number with leading zeros to reach the specified length.
    """
    number_str = str(number)
    return f"{'0' * (max_length - len(number_str))}{number_str}"


@beartype
def verify_variable_order(data: pl.DataFrame) -> None:
    """Verify that sequenceId and subsequenceId are sorted."""
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


def format_delta(time_delta):
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
    """
    Get probabilities and predictions for autoregression using a Polars-native loop.
    This function now also returns the corresponding sequence IDs for each prediction.
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
            data_subset,  # numpy_to_pytorch now handles polars DF
            column_types,
            apply_normalization_inversion=False,
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
            f"subseq-id: {subsequence_id}: total: {format_delta(t4-t0)}s - {format_delta(t1 - t0)}s - {format_delta(t2 - t1)}s - {format_delta(t3 - t2)}s - {format_delta(t4 - t3)}s"
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
    @beartype
    def __init__(
        self,
        model_path: str,
        project_path: str,
        id_maps: Optional[dict[str, dict[Union[str, int], int]]],
        selected_columns_statistics: dict[str, dict[str, float]],
        map_to_id: bool,
        categorical_columns: list[str],
        real_columns: list[str],
        selected_columns: Optional[list[str]],
        target_columns: list[str],
        target_column_types: dict[str, str],
        sample_from_distribution_columns: Optional[list[str]],
        infer_with_dropout: bool,
        inference_batch_size: int,
        device: str,
        args_config: dict[str, Any],
        training_config_path: str,
    ):
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
        self.selected_columns = selected_columns
        self.target_columns = target_columns
        self.target_column_types = target_column_types
        self.sample_from_distribution_columns = sample_from_distribution_columns
        self.infer_with_dropout = infer_with_dropout
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
        """
        Invert the normalization of values for a target column.

        Args:
            values: Normalized values.
            target_column: Target column name.

        Returns:
            Denormalized values.
        """
        std = self.selected_columns_statistics[target_column]["std"]
        mean = self.selected_columns_statistics[target_column]["mean"]
        return (values * (std - 1e-9)) + mean

    @beartype
    def infer(
        self,
        x: Optional[dict[str, np.ndarray]],
        probs: Optional[dict[str, np.ndarray]] = None,
        return_probs: bool = False,
        apply_normalization_inversion: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Perform inference on the input data.

        Args:
            x: Input data.
            probs: Pre-computed probabilities (optional).
            return_probs: Whether to return probabilities.

        Returns:
            Dictionary of inference results.
        """
        if probs is None or (
            x is not None and len(set(x.keys()).difference(set(probs.keys()))) > 0
        ):  # type: ignore
            assert x is not None
            size = x[self.target_columns[0]].shape[0]
            if (
                probs is not None
                and len(set(x.keys()).difference(set(probs.keys()))) > 0
            ):  # type: ignore
                assert x is not None
                warnings.warn(
                    f"Not all keys in x are in probs - {x.keys() = } != {probs.keys() = }. Full inference is executed."
                )

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
                    )[:size, :]
                    for target_column in self.target_columns
                }
            elif self.inference_model_type == "pt":
                assert x is not None
                x_adjusted = self.prepare_inference_batches(x, pad_to_batch_size=False)
                outs = infer_with_model(
                    self.inference_model,
                    x_adjusted,
                    self.device,
                    size,
                    self.target_columns,
                )
            else:
                assert False
                outs = {}  # for type checking

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

        if apply_normalization_inversion:
            for target_column, output in outs.items():
                if self.target_column_types[target_column] == "real":
                    outs[target_column] = self.invert_normalization(
                        output, target_column
                    )
        return outs

    @beartype
    def prepare_inference_batches(
        self, x: dict[str, np.ndarray], pad_to_batch_size: bool
    ) -> list[dict[str, np.ndarray]]:
        size = x[self.target_columns[0]].shape[0]
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
        """
        Perform pure inference using ONNX session.

        Args:
            x: Input data.

        Returns:
            List of output arrays.
        """
        ort_inputs = {
            session_input.name: self.expand_to_batch_size(x[col])
            for session_input, col in zip(
                self.ort_session.get_inputs(),
                self.categorical_columns + self.real_columns,
            )
        }
        ort_outs = self.ort_session.run(None, ort_inputs)
        return ort_outs

    @beartype
    def expand_to_batch_size(self, x: np.ndarray) -> np.ndarray:
        """
        Expand input to match the inference batch size.

        Args:
            x: Input array.

        Returns:
            Expanded array.
        """
        repetitions = self.inference_batch_size // x.shape[0]
        filler = self.inference_batch_size % x.shape[0]
        return np.concatenate(([x] * repetitions) + [x[0:filler, :]], axis=0)


@beartype
def normalize(outs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Normalize the output probabilities.

    Args:
        outs: Dictionary of output arrays.

    Returns:
        Dictionary of normalized probabilities.
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
    """
    Sample from cumulative sum of probabilities.

    Args:
        probs: Probability array.

    Returns:
        Sampled indices.
    """
    cumulative_probs = np.cumsum(np.exp(probs), axis=1)
    random_threshold = np.random.rand(cumulative_probs.shape[0], 1)
    random_threshold = np.repeat(random_threshold, probs.shape[1], axis=1)
    return (random_threshold < cumulative_probs).argmax(axis=1)
