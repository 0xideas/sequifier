import json
import os
from typing import Optional, Union

import numpy as np
import yaml
from beartype import beartype
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from sequifier.helpers import (
    ModelWindowView,
    StoredWindowLayout,
    canonicalize_polars_dtype_name,
    normalize_path,
    resolve_window_view,
    stored_window_layout_from_metadata,
    try_catch_excess_keys,
)
from sequifier.special_tokens import validate_special_token_ids


@beartype
def load_inferer_config(
    config_path: str, args_config: dict, skip_metadata: bool
) -> "InfererModel":
    """Load inference YAML plus CLI overrides and optional metadata fields."""
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

        validate_special_token_ids(
            metadata_config["special_token_ids"],
            source=f"metadata config '{metadata_config_path}'",
        )
        storage_layout = stored_window_layout_from_metadata(metadata_config)
        if storage_layout.version != 2:
            raise ValueError(
                "Inference requires metadata stored_window_layout_version=2, "
                f"got {storage_layout.version}."
            )
        training_objective = config_values["training_objective"]
        target_offset = (
            0
            if training_objective == "bert"
            else int(config_values.pop("target_offset", 1))
        )
        window_view = ModelWindowView(
            context_length=int(config_values.pop("context_length")),
            objective=training_objective,
            target_offset=target_offset,
        )
        resolve_window_view(storage_layout, window_view)
        config_values["storage_layout"] = storage_layout
        config_values["window_view"] = window_view
        for key in (
            "target_offset",
            "stored_context_width",
            "max_target_offset",
            "stored_window_layout_version",
        ):
            config_values.pop(key, None)

        config_values["column_types"] = config_values.get(
            "column_types", metadata_config["column_types"]
        )

        if config_values["input_columns"] is None:
            config_values["input_columns"] = list(config_values["column_types"].keys())

        configured_column_types = config_values["column_types"]

        config_values["categorical_columns"] = [
            col
            for col, type_ in configured_column_types.items()
            if "int" in type_.lower() and col in config_values["input_columns"]
        ]
        config_values["real_columns"] = [
            col
            for col, type_ in configured_column_types.items()
            if "float" in type_.lower() and col in config_values["input_columns"]
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
    """Top-level inference config."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    project_root: str
    metadata_config_path: str
    model_path: Union[str, list[str]]
    model_type: str
    training_objective: str
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

    enforce_determinism: bool = Field(default=False)
    output_probabilities: bool = Field(default=False)
    map_to_id: bool = Field(default=True)
    seed: int
    device: str
    storage_layout: StoredWindowLayout
    window_view: ModelWindowView
    prediction_length: Optional[int] = None
    inference_batch_size: int

    sample_from_distribution_columns: Optional[list[str]] = Field(default=None)
    infer_with_dropout: bool = Field(default=False)
    autoregression: bool = Field(default=False)
    autoregression_total_steps: Optional[int] = Field(default=None)

    @model_validator(mode="after")
    def normalize_prediction_length(self):
        if self.window_view.objective != self.training_objective:
            raise ValueError(
                "window_view objective must match training_objective "
                f"({self.window_view.objective} != {self.training_objective})."
            )
        if self.prediction_length is None:
            self.prediction_length = (
                self.window_view.context_length
                if self.training_objective == "bert"
                else 1
            )
        if self.training_objective == "bert":
            if self.prediction_length != self.window_view.context_length:
                raise ValueError(
                    "For BERT inference, prediction_length must be equal to context_length "
                    f"(got prediction_length={self.prediction_length}, context_length={self.window_view.context_length})."
                )
        else:
            resolve_window_view(self.storage_layout, self.window_view)
        return self

    @field_validator("training_objective")
    @classmethod
    def validate_training_objective(cls, v):
        if v not in ["causal", "bert", "final_value", "next_occurrence"]:
            raise ValueError(
                "Only 'causal', 'bert', 'final_value', and 'next_occurrence' "
                f"are allowed, found {v}"
            )
        return v

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

    @field_validator("autoregression_total_steps")
    @classmethod
    def validate_autoregression_total_steps(
        cls, v: Optional[int], info: ValidationInfo
    ) -> Optional[int]:
        if v is None and info.data.get("autoregression") is True:
            raise ValueError(
                "If autoregression==True, 'autoregression_total_steps' needs to be set to an integer value."
            )
        if v is not None and v < 1:
            raise ValueError("autoregression_total_steps must by >= 1.")
        if v is not None and v > 1:
            if not info.data.get("autoregression"):
                raise ValueError(
                    f"'autoregression_total_steps' can only be larger than 1 if 'autoregression' is true: {info.data.get('autoregression')}"
                )

            if not np.all(
                np.array(sorted(info.data.get("input_columns")))
                == np.array(sorted(info.data.get("target_columns")))
            ):
                raise ValueError(
                    "'autoregression_total_steps' can only be larger than 1 if 'input_columns' and 'target_columns' are identical"
                )

        return v

    @field_validator("autoregression")
    @classmethod
    def validate_autoregression(cls, v: bool, info: ValidationInfo):
        if v and info.data.get("model_type") == "embedding":
            raise ValueError("Autoregression is not possible for embedding models")
        if (
            v
            and info.data.get("prediction_length") is not None
            and info.data.get("prediction_length") > 1
        ):
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

        if (
            v
            and info.data.get("training_objective") is not None
            and info.data.get("training_objective") == "bert"
        ):
            raise ValueError(
                "Autoregressive inference is not possible with BERT-style models."
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

    @field_validator("column_types")
    @classmethod
    def validate_column_types(cls, v: dict, info: ValidationInfo) -> dict:
        normalized = {
            column: canonicalize_polars_dtype_name(dtype) for column, dtype in v.items()
        }
        input_columns = info.data.get("input_columns", [])
        missing_input_columns = [
            column for column in input_columns if column not in normalized
        ]
        if missing_input_columns:
            raise ValueError(
                "column_types must include every input column. "
                f"Missing: {missing_input_columns}"
            )
        return normalized

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

    def __init__(self, **data):
        super().__init__(**data)
        column_ordered = list(self.column_types.keys())
        columns_ordered_filtered = [
            c for c in column_ordered if c in self.target_columns
        ]
        if not (columns_ordered_filtered == self.target_columns):
            raise ValueError(f"{columns_ordered_filtered} != {self.target_columns}")
