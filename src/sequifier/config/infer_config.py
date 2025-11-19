import json
import os
from typing import Optional, Union

import numpy as np
import yaml
from beartype import beartype
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from sequifier.helpers import normalize_path, try_catch_excess_keys


@beartype
def load_inferer_config(
    config_path: str, args_config: dict, skip_metadata: bool
) -> "InfererModel":
    """
    Load inferer configuration from a YAML file and update it with args_config.

    Args:
        config_path: Path to the YAML configuration file.
        args_config: Dictionary containing additional configuration arguments.
        skip_metadata: Flag indicating whether to process the configuration or not.

    Returns:
        InfererModel instance with loaded configuration.
    """
    with open(config_path, "r") as f:
        config_values = yaml.safe_load(f)
    config_values.update(args_config)

    config_values["seed"] = config_values.get("seed", 1010)

    if not skip_metadata:
        metadata_config_path = config_values.get("metadata_config_path")

        with open(
            normalize_path(metadata_config_path, config_values["project_root"]), "r"
        ) as f:
            metadata_config = json.load(f)

        config_values["column_types"] = config_values.get(
            "column_types", metadata_config["column_types"]
        )

        if config_values["input_columns"] is None:
            config_values["input_columns"] = list(config_values["column_types"].keys())

        config_values["categorical_columns"] = [
            col
            for col, type_ in metadata_config["column_types"].items()
            if "int64" in type_.lower() and col in config_values["input_columns"]
        ]
        config_values["real_columns"] = [
            col
            for col, type_ in metadata_config["column_types"].items()
            if "float64" in type_.lower() and col in config_values["input_columns"]
        ]
        if not (
            len(config_values["real_columns"] + config_values["categorical_columns"])
            > 0
        ):
            raise ValueError("No columns found in config")
        config_values["data_path"] = normalize_path(
            config_values.get(
                "data_path",
                metadata_config["split_paths"][
                    min(2, len(metadata_config["split_paths"]) - 1)
                ],
            ),
            config_values["project_root"],
        )

    return try_catch_excess_keys(config_path, InfererModel, config_values)


class InfererModel(BaseModel):
    """Pydantic model for inference configuration.

    Attributes:
        project_root: The path to the sequifier project directory.
        metadata_config_path: The path to the data-driven configuration file.
        model_path: The path to the trained model file(s).
        model_type: The type of model, either 'embedding' or 'generative'.
        data_path: The path to the data to be used for inference.
        training_config_path: The path to the training configuration file.
        read_format: The file format of the input data (e.g., 'csv', 'parquet').
        write_format: The file format for the inference output.
        input_columns: The list of input columns used for inference.
        categorical_columns: A list of columns that are categorical.
        real_columns: A list of columns that are real-valued.
        target_columns: The list of target columns for inference.
        column_types: A dictionary mapping each column to its numeric type ('int64' or 'float64').
        target_column_types: A dictionary mapping target columns to their types ('categorical' or 'real').
        output_probabilities: If True, outputs the probability distributions for categorical target columns.
        map_to_id: If True, maps categorical output values back to their original IDs.
        seed: The random seed for reproducibility.
        device: The device to run inference on (e.g., 'cuda', 'cpu', 'mps').
        seq_length: The sequence length of the model's input.
        inference_batch_size: The batch size for inference.
        distributed: If True, enables distributed inference.
        load_full_data_to_ram: If True, loads the entire dataset into RAM.
        world_size: The number of processes for distributed inference.
        num_workers: The number of worker threads for data loading.
        sample_from_distribution_columns: A list of columns from which to sample from the distribution.
        infer_with_dropout: If True, applies dropout during inference.
        autoregression: If True, performs autoregressive inference.
        autoregression_extra_steps: The number of additional steps for autoregressive inference.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    project_root: str
    metadata_config_path: str
    model_path: Union[str, list[str]]
    model_type: str
    data_path: str
    training_config_path: str = Field(default="configs/train.yaml")
    read_format: str = Field(default="parquet")
    write_format: str = Field(default="csv")

    input_columns: list[str]
    categorical_columns: list[str]
    real_columns: list[str]
    target_columns: list[str]
    column_types: dict[str, str]
    target_column_types: dict[str, str]

    output_probabilities: bool = Field(default=False)
    map_to_id: bool = Field(default=True)
    seed: int
    device: str
    seq_length: int
    prediction_length: int = 1
    inference_batch_size: int

    distributed: bool = False
    load_full_data_to_ram: bool = True
    world_size: int = 1
    num_workers: int = 0

    sample_from_distribution_columns: Optional[list[str]] = Field(default=None)
    infer_with_dropout: bool = Field(default=False)
    autoregression: bool = Field(default=False)
    autoregression_extra_steps: Optional[int] = Field(default=None)

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        if v not in [
            "embedding",
            "generative",
        ]:
            raise ValueError(
                f"model_type must be one of 'embedding' and 'generative, {v} isn't"
            )
        return v

    @field_validator("output_probabilities")
    @classmethod
    def validate_output_probabilities(cls, v: str, info: ValidationInfo) -> str:
        if v and info.data.get("model_type") == "embedding":
            raise ValueError(
                "For embedding models, 'output_probabilities' must be set to false"
            )
        return v

    @field_validator("training_config_path")
    @classmethod
    def validate_training_config_path(cls, v: str) -> str:
        if not (v is None or os.path.exists(v)):
            raise ValueError(f"{v} does not exist")
        return v

    @field_validator("autoregression_extra_steps")
    @classmethod
    def validate_autoregression_extra_steps(cls, v: bool, info: ValidationInfo) -> bool:
        if v is not None and v > 0:
            if not info.data.get("autoregression"):
                raise ValueError(
                    f"'autoregression_extra_steps' can only be larger than 0 if 'autoregression' is true: {info.data.get('autoregression')}"
                )

            if not np.all(
                np.array(sorted(info.data.get("input_columns")))
                == np.array(sorted(info.data.get("target_columns")))
            ):
                raise ValueError(
                    "'autoregression_extra_steps' can only be larger than 0 if 'input_columns' and 'target_columns' are identical"
                )

        return v

    @field_validator("autoregression")
    @classmethod
    def validate_autoregression(cls, v: bool, info: ValidationInfo):
        if v and info.data.get("model_type") == "embedding":
            raise ValueError("Autoregression is not possible for embedding models")
        if v and info.data.get("prediction_length") > 1:
            raise ValueError(
                "Autoregressive inference is not possible for models with prediction_length > 1"
            )
        if v and not np.all(
            np.array(sorted(info.data.get("input_columns")))
            == np.array(sorted(info.data.get("target_columns")))
        ):
            raise ValueError(
                "Autoregressive inference with non-identical 'input_columns' and 'target_columns' is possible but should not be performed"
            )
        return v

    @field_validator("data_path")
    @classmethod
    def validate_data_path(cls, v: str, info: ValidationInfo) -> str:
        if isinstance(v, str):
            v2 = normalize_path(v, info.data.get("project_root"))
            if not os.path.exists(v2):
                raise ValueError(f"{v2} does not exist")
        if isinstance(v, list):
            for vv in v:
                v2 = normalize_path(v, info.data.get("project_root"))
                if not os.path.exists(v2):
                    raise ValueError(f"{v2} does not exist")
        return v

    @field_validator("read_format", "write_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if v not in ["csv", "parquet", "pt"]:
            raise ValueError("Currently only 'csv', 'parquet' and 'pt' are supported")
        return v

    @field_validator("target_column_types")
    @classmethod
    def validate_target_column_types(cls, v: dict, info: ValidationInfo) -> dict:
        if not all(vv in ["categorical", "real"] for vv in v.values()):
            raise ValueError(
                "Target column types must be either 'categorical' or 'real'"
            )
        if list(v.keys()) != info.data.get("target_columns", []):
            raise ValueError(
                "target_columns and target_column_types must contain the same keys in the same order"
            )
        return v

    @field_validator("map_to_id")
    @classmethod
    def validate_map_to_id(cls, v: bool, info: ValidationInfo) -> bool:
        if v and not any(
            vv == "categorical"
            for vv in info.data.get("target_column_types", {}).values()
        ):
            raise ValueError(
                "map_to_id can only be True if at least one target variable is categorical"
            )
        return v

    @field_validator("distributed")
    @classmethod
    def validate_distributed_inference(cls, v: bool, info: ValidationInfo) -> bool:
        if v and info.data.get("read_format") != "pt":
            raise ValueError(
                "Distributed inference is only supported for preprocessed '.pt' files. Please set read_format to 'pt'."
            )
        return v

    def __init__(self, **data):
        super().__init__(**data)
        column_ordered = list(self.column_types.keys())
        columns_ordered_filtered = [
            c for c in column_ordered if c in self.target_columns
        ]
        if not (columns_ordered_filtered == self.target_columns):
            raise ValueError(f"{columns_ordered_filtered} != {self.target_columns}")
