import json
import os
from typing import Optional, Union

import yaml
from beartype import beartype
from pydantic import BaseModel, Field, validator

from sequifier.helpers import normalize_path


@beartype
def load_inferer_config(
    config_path: str, args_config: dict, on_unprocessed: bool
) -> "InfererModel":
    """
    Load inferer configuration from a YAML file and update it with args_config.

    Args:
        config_path: Path to the YAML configuration file.
        args_config: Dictionary containing additional configuration arguments.
        on_unprocessed: Flag indicating whether to process the configuration or not.

    Returns:
        InfererModel instance with loaded configuration.
    """
    with open(config_path, "r") as f:
        config_values = yaml.safe_load(f)
    config_values.update(args_config)

    if not on_unprocessed:
        ddconfig_path = config_values.get("ddconfig_path")

        with open(
            normalize_path(ddconfig_path, config_values["project_path"]), "r"
        ) as f:
            dd_config = json.load(f)

        config_values["column_types"] = config_values.get(
            "column_types", dd_config["column_types"]
        )

        if config_values["selected_columns"] is None:
            config_values["selected_columns"] = list(
                config_values["column_types"].keys()
            )

        config_values["categorical_columns"] = [
            col
            for col, type_ in dd_config["column_types"].items()
            if type_ == "int64" and col in config_values["selected_columns"]
        ]
        config_values["real_columns"] = [
            col
            for col, type_ in dd_config["column_types"].items()
            if type_ == "float64" and col in config_values["selected_columns"]
        ]
        config_values["data_path"] = normalize_path(
            config_values.get(
                "data_path",
                dd_config["split_paths"][min(2, len(dd_config["split_paths"]) - 1)],
            ),
            config_values["project_path"],
        )

    return InfererModel(**config_values)


class InfererModel(BaseModel):
    project_path: str
    ddconfig_path: str
    model_path: Union[str, list[str]]
    data_path: str
    training_config_path: str = Field(default="configs/train.yaml")
    read_format: str = Field(default="parquet")
    write_format: str = Field(default="csv")

    selected_columns: list[str]
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
    inference_batch_size: int

    sample_from_distribution_columns: Optional[list[str]] = Field(default=None)
    infer_with_dropout: bool = Field(default=False)
    autoregression: bool = Field(default=True)
    autoregression_additional_steps: Optional[int] = Field(default=None)

    @validator("training_config_path")
    def validate_training_config_path(cls, v: str) -> str:
        if not (v is None or os.path.exists(v)):
            raise ValueError(f"{v} does not exist")
        return v

    @validator("data_path")
    def validate_data_path(cls, v: str, values: dict) -> str:
        if isinstance(v, str):
            v2 = normalize_path(v, values["project_path"])
            if not os.path.exists(v2):
                raise ValueError(f"{v2} does not exist")
        if isinstance(v, list):
            for vv in v:
                v2 = normalize_path(v, values["project_path"])
                if not os.path.exists(v2):
                    raise ValueError(f"{v2} does not exist")
        return v

    @validator("read_format", "write_format")
    def validate_format(cls, v: str) -> str:
        if v not in ["csv", "parquet"]:
            raise ValueError("Currently only 'csv' and 'parquet' are supported")
        return v

    @validator("target_column_types")
    def validate_target_column_types(cls, v: dict, values: dict) -> dict:
        if not all(vv in ["categorical", "real"] for vv in v.values()):
            raise ValueError(
                "Target column types must be either 'categorical' or 'real'"
            )
        if list(v.keys()) != values.get("target_columns", []):
            raise ValueError(
                "target_columns and target_column_types must contain the same keys in the same order"
            )
        return v

    @validator("map_to_id")
    def validate_map_to_id(cls, v: bool, values: dict) -> bool:
        if v and not any(
            vv == "categorical" for vv in values.get("target_column_types", {}).values()
        ):
            raise ValueError(
                "map_to_id can only be True if at least one target variable is categorical"
            )
        return v

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        column_ordered = list(self.column_types.keys())
        columns_ordered_filtered = [
            c for c in column_ordered if c in self.target_columns
        ]
        assert (
            columns_ordered_filtered == self.target_columns
        ), f"{columns_ordered_filtered} != {self.target_columns}"
