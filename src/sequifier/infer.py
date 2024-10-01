import json
import os
import warnings
from typing import Any, Optional, Union
from warnings import simplefilter

import numpy as np
import onnxruntime
import pandas as pd
import torch
from beartype import beartype

from sequifier.config.infer_config import load_inferer_config
from sequifier.helpers import (
    PANDAS_TO_TORCH_TYPES,
    construct_index_maps,
    normalize_path,
    numpy_to_pytorch,
    read_data,
    subset_to_selected_columns,
    write_data,
)
from sequifier.train import infer_with_model, load_inference_model

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


@beartype
def infer(args: Any, args_config: dict[str, Any]) -> None:
    config_path = (
        args.config_path if args.config_path is not None else "configs/infer.yaml"
    )

    config = load_inferer_config(config_path, args_config, args.on_unprocessed)

    if config.map_to_id or (len(config.real_columns) > 0):
        assert config.ddconfig_path is not None, (
            "If you want to map to id, you need to provide a file path to a json that contains: {{'id_maps':{...}}} to ddconfig_path"
            "\nIf you have real columns in the data, you need to provide a json that contains: {{'min_max_values':{COL_NAME:{'min':..., 'max':...}}}}"
        )
        with open(normalize_path(config.ddconfig_path, config.project_path), "r") as f:
            dd_config = json.loads(f.read())
            id_maps = dd_config["id_maps"]
            min_max_values = dd_config["min_max_values"]
    else:
        id_maps = None
        min_max_values = {}

    inferer = Inferer(
        config.model_path,
        config.project_path,
        id_maps,
        min_max_values,
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

    model_id = os.path.split(config.model_path)[1].replace(
        f".{inferer.inference_model_type}", ""
    )

    print(f"Inferring for {model_id}")
    data = read_data(config.data_path, config.read_format)

    if config.selected_columns is not None:
        data = subset_to_selected_columns(data, config.selected_columns)

    if not config.autoregression:
        probs, preds = get_probs_preds(config, inferer, data, column_types)
    else:
        if config.autoregression_additional_steps is not None:
            data = expand_data_by_autoregression(
                data, config.autoregression_additional_steps, config.seq_length
            )

        probs, preds = get_probs_preds_autoregression(
            config, inferer, data, column_types, config.seq_length
        )

    if inferer.map_to_id:
        for target_column, predictions in preds.items():
            if target_column in inferer.index_map:
                preds[target_column] = np.array(
                    [inferer.index_map[target_column][i] for i in predictions]
                )

    os.makedirs(
        os.path.join(config.project_path, "outputs", "predictions"), exist_ok=True
    )

    if config.output_probabilities:
        assert probs is not None
        os.makedirs(
            os.path.join(config.project_path, "outputs", "probabilities"), exist_ok=True
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
                write_data(
                    pd.DataFrame(probs[target_column]),
                    probabilities_path,
                    config.write_format,
                )
    n_input_cols = len(np.unique(data["inputCol"]))
    predictions = pd.DataFrame(
        {
            **{"sequenceId": list(data["sequenceId"].values)[::n_input_cols]},
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
    data: pd.DataFrame, autoregression_additional_steps: int, seq_length: int
) -> pd.DataFrame:
    """
    Expand data for autoregression by adding additional steps.

    Args:
        data: Input DataFrame.
        autoregression_additional_steps: Number of additional steps for autoregression.
        seq_length: Length of the sequence.

    Returns:
        Expanded DataFrame for autoregression.
    """
    verify_variable_order(data)

    data_cols = [str(c) for c in range(seq_length, 0, -1)]

    autoregression_additional_observations = []
    for sequence_id, sequence_data in data.groupby("sequenceId"):
        max_subsequence_id = sequence_data["subsequenceId"].values.max()
        last_observation = sequence_data.query(f"subsequenceId=={max_subsequence_id}")

        for offset in range(1, autoregression_additional_steps + 1):
            sequence_id_fields = np.repeat(sequence_id, last_observation.shape[0])
            subsequence_id_fields = np.repeat(
                max_subsequence_id + offset, last_observation.shape[0]
            )
            input_col_fields = last_observation["inputCol"].values
            metadata = pd.DataFrame(
                {
                    "sequenceId": sequence_id_fields,
                    "subsequenceId": subsequence_id_fields,
                    "inputCol": input_col_fields,
                }
            )

            empty_data_fields = (
                np.ones((last_observation.shape[0], min(seq_length, offset))) * np.inf
            )
            offset_data_fields = last_observation[data_cols].values[
                :, min(offset, last_observation.shape[1]) :
            ]
            data_fields = np.concatenate(
                [offset_data_fields, empty_data_fields], axis=1
            )
            data_df = pd.DataFrame(data_fields, columns=data_cols)
            observation = pd.concat([metadata, data_df], axis=1)
            autoregression_additional_observations.append(observation)

    data = pd.concat([data] + autoregression_additional_observations, axis=0)

    return data.sort_values(
        ["sequenceId", "subsequenceId"], ascending=True
    ).reset_index(drop=True)


@beartype
def get_probs_preds(
    config: Any,
    inferer: "Inferer",
    data: pd.DataFrame,
    column_types: dict[str, torch.dtype],
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
        preds = inferer.infer(None, probs)
    else:
        probs = None
        preds = inferer.infer(X)

    return (probs, preds)


@beartype
def fill_in_predictions(
    data: pd.DataFrame,
    sequence_id_to_subsequence_ids: dict[int, np.ndarray],
    ids_to_row: dict[str, int],
    sequence_ids: np.ndarray,
    subsequence_id: int,
    preds: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Fill in predictions for the given data.

    Args:
        data: Input DataFrame.
        sequence_id_to_subsequence_ids: Mapping of sequence IDs to subsequence IDs.
        ids_to_row: Mapping of IDs to row indices.
        sequence_ids: Array of sequence IDs.
        subsequence_id: Current subsequence ID.
        preds: Dictionary of predictions.

    Returns:
        Updated DataFrame with filled-in predictions.
    """
    sequence_ids_distinct = sorted(list(np.unique(sequence_ids)))

    for input_col, preds_vals in preds.items():
        assert len(preds_vals) == len(sequence_ids_distinct)
        for sequence_id, pred in zip(sequence_ids_distinct, preds_vals.flatten()):
            sequence_id_subsequence_ids = sequence_id_to_subsequence_ids[sequence_id]
            sequence_id_subsequence_ids = sequence_id_subsequence_ids[
                sequence_id_subsequence_ids > subsequence_id
            ]
            for subsequence_id2 in sequence_id_subsequence_ids:
                offset = subsequence_id2 - subsequence_id
                assert offset > 0
                i = ids_to_row[f"{sequence_id}-{subsequence_id2}-{input_col}"]
                data.loc[i, str(offset)] = pred

    return data


@beartype
def fill_number(number: Union[int, float], max_length: int) -> str:
    """
    Fill a number with leading zeros to reach the specified length.

    Args:
        number: Number to be filled.
        max_length: Maximum length of the resulting string.

    Returns:
        String representation of the number with leading zeros.
    """
    number_str = str(number)
    return f"{'0' * (max_length - len(number_str))}{number_str}"


@beartype
def verify_variable_order(data: pd.DataFrame) -> None:
    sequence_ids = data["sequenceId"].values
    assert np.all(
        sequence_ids[1:] - sequence_ids[:-1] >= 0
    ), "sequenceId must be in ascending order for autoregression"

    for _, sequence_id_group in data[["sequenceId", "subsequenceId"]].groupby(
        "sequenceId"
    ):
        subsequence_ids = sequence_id_group["subsequenceId"].values
        assert np.all(
            subsequence_ids[1:] - subsequence_ids[:-1] >= 0
        ), "subsequenceId must be in ascending order for autoregression"


@beartype
def get_probs_preds_autoregression(
    config: Any,
    inferer: "Inferer",
    data: pd.DataFrame,
    column_types: dict[str, torch.dtype],
    seq_length: int,
) -> tuple[Optional[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """
    Get probabilities and predictions for autoregression.

    Args:
        config: Configuration object.
        inferer: Inferer object.
        data: Input DataFrame.
        column_types: Dictionary of column types.
        seq_length: Length of the sequence.

    Returns:
        Tuple of probabilities and predictions.
    """
    sequence_ids = data["sequenceId"].values
    verify_variable_order(data)

    sequence_id_to_min_subsequence_id = (
        data[["sequenceId", "subsequenceId"]]
        .groupby("sequenceId")
        .agg({"subsequenceId": "min"})
        .to_dict()["subsequenceId"]
    )

    data["subsequenceIdAdjusted"] = [
        row["subsequenceId"] - sequence_id_to_min_subsequence_id[row["sequenceId"]]
        for _, row in data[["sequenceId", "subsequenceId"]].iterrows()
    ]

    sequence_id_to_subsequence_ids = {
        sequence_id_: np.array(subsequence_ids_)
        for sequence_id_, subsequence_ids_ in data[
            ["sequenceId", "subsequenceIdAdjusted"]
        ]
        .groupby("sequenceId")
        .agg({"subsequenceIdAdjusted": lambda x: sorted(list(set(x)))})
        .to_dict()["subsequenceIdAdjusted"]
        .items()
    }

    ids_to_row = {
        f"{row['sequenceId']}-{row['subsequenceIdAdjusted']}-{row['inputCol']}": i
        for i, row in data.iterrows()
    }

    preds_list, probs_list, sort_keys = [], [], []
    subsequence_ids_distinct = sorted(list(np.unique(data["subsequenceIdAdjusted"])))
    subsequence_ids = data["subsequenceIdAdjusted"].values
    max_length = len(str(np.max(subsequence_ids_distinct)))
    for subsequence_id in subsequence_ids_distinct:
        subsequence_filter = subsequence_ids == subsequence_id
        data_subset = data.loc[subsequence_filter, :]
        sequence_ids_present = sequence_ids[subsequence_filter]

        sort_keys.extend(
            [
                f"{fill_number(int(seq_id), max_length)}-{fill_number(int(subsequence_id), max_length)}"
                for seq_id in np.unique(sequence_ids_present)
            ]
        )

        probs, preds = get_probs_preds(config, inferer, data_subset, column_types)
        preds_list.append(preds)
        if probs is not None:
            probs_list.append(probs)

        data = fill_in_predictions(
            data,
            sequence_id_to_subsequence_ids,
            ids_to_row,
            sequence_ids_present,
            int(subsequence_id),
            preds,
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
    return probs, preds


class Inferer:
    @beartype
    def __init__(
        self,
        model_path: str,
        project_path: str,
        id_maps: Optional[dict[str, dict[Union[str, int], int]]],
        min_max_values: dict[str, dict[str, float]],
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
        self.min_max_values = min_max_values
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
        min_ = self.min_max_values[target_column]["min"]
        max_ = self.min_max_values[target_column]["max"]
        return np.array(
            [(((v + 0.8) / 1.6) * (max_ - min_)) + min_ for v in values.flatten()]
        ).reshape(*values.shape)

    @beartype
    def infer(
        self,
        x: Optional[dict[str, np.ndarray]],
        probs: Optional[dict[str, np.ndarray]] = None,
        return_probs: bool = False,
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

        for target_column, output in outs.items():
            if self.target_column_types[target_column] == "real":
                outs[target_column] = self.invert_normalization(output, target_column)
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
