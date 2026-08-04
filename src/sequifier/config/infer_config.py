import os
from typing import Generic, Optional, TypeVar, Union

import numpy as np
from beartype import beartype
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from sequifier.config.composition import (
    load_composed_yaml_config,
    merge_config_fragments,
)
from sequifier.config.metadata import (
    DatasetMetadata,
    extract_inline_metadata,
    load_dataset_metadata,
)
from sequifier.helpers import (
    ModelWindowView,
    StoredWindowLayout,
    canonicalize_polars_dtype_name,
    derive_target_column_types,
    metadata_config_path_from_preprocessing_data_path,
    normalize_path,
    resolve_window_view,
    try_catch_excess_keys,
)
from sequifier.objectives import (
    ALLOWED_OBJECTIVE_NAMES,
    OBJECTIVE_NAME_MESSAGE,
    get_objective_class,
    target_offset_for_objective,
)


@beartype
def load_inferer_config(
    config_path: str, args_config: dict, skip_metadata: bool
) -> "ResolvedInferenceConfig":
    """Compose and validate inference YAML, then resolve dataset metadata."""

    config_values = load_composed_yaml_config(config_path)

    cli_values = {
        key: value for key, value in args_config.items() if key != "skip_metadata"
    }
    authored_values = merge_config_fragments((config_values, cli_values))
    authored_values, extracted_metadata_values = extract_inline_metadata(
        authored_values
    )
    inline_metadata_values = extracted_metadata_values if skip_metadata else None
    config = try_catch_excess_keys(config_path, InferenceConfig, authored_values)

    metadata_path = _effective_metadata_config_path(config)
    if skip_metadata:
        if inline_metadata_values is None:
            raise ValueError(
                "skip_metadata requires inline storage_layout and column values "
                "so the inference config can still be resolved."
            )
        metadata = DatasetMetadata.model_validate(inline_metadata_values)
    else:
        if metadata_path is None:
            raise ValueError(
                f"Inference config '{config_path}' must define metadata_config_path "
                "or preprocessing_data_path when metadata loading is enabled."
            )
        metadata = load_dataset_metadata(
            normalize_path(metadata_path, config.project_root)
        )

    return resolve_inference_config(config, metadata)


def _effective_metadata_config_path(config: "InferenceConfig") -> Optional[str]:
    if config.metadata_config_path:
        return config.metadata_config_path
    if config.preprocessing_data_path:
        return metadata_config_path_from_preprocessing_data_path(
            config.preprocessing_data_path
        )
    return None


def resolve_inference_config(
    config: "InferenceConfig", metadata: DatasetMetadata
) -> "ResolvedInferenceConfig":
    """Return an inference config with all metadata-derived values populated."""

    storage_layout = metadata.storage_layout
    if storage_layout.version != 2:
        raise ValueError(
            "Inference requires metadata stored_window_layout_version=2, "
            f"got {storage_layout.version}."
        )
    column_data_types = config.column_data_types or metadata.column_data_types
    input_columns = (
        list(column_data_types)
        if config.input_columns is None
        else config.input_columns
    )
    categorical_columns = [
        column
        for column, type_name in column_data_types.items()
        if "int" in type_name.lower() and column in input_columns
    ]
    real_columns = [
        column
        for column, type_name in column_data_types.items()
        if "float" in type_name.lower() and column in input_columns
    ]
    if not categorical_columns and not real_columns:
        raise ValueError("No columns found in resolved inference config")

    target_column_types = config.target_column_types or derive_target_column_types(
        config.target_columns, column_data_types
    )
    window_view = ModelWindowView(
        context_length=config.context_length,
        objective=config.training_objective,
        target_offset=target_offset_for_objective(
            config.training_objective, config.target_offset
        ),
    )
    resolve_window_view(storage_layout, window_view)

    if config.data_path is None and not metadata.split_paths:
        raise ValueError(
            "Resolved inference config needs data_path when metadata does not "
            "provide split_paths."
        )
    data_path = (
        config.data_path or metadata.split_paths[min(2, len(metadata.split_paths) - 1)]
    )
    values = config.model_dump(mode="python")
    values.update(
        {
            "metadata_config_path": _effective_metadata_config_path(config),
            "data_path": normalize_path(data_path, config.project_root),
            "input_columns": input_columns,
            "column_data_types": column_data_types,
            "categorical_columns": categorical_columns,
            "real_columns": real_columns,
            "target_column_types": target_column_types,
            "storage_layout": storage_layout,
            "window_view": window_view,
            "dataset_metadata": metadata,
        }
    )
    return ResolvedInferenceConfig.model_validate(values)


_PathT = TypeVar("_PathT")
_InputColumnsT = TypeVar("_InputColumnsT")
_ColumnTypesT = TypeVar("_ColumnTypesT")


class _InferenceConfigBase(BaseModel, Generic[_PathT, _InputColumnsT, _ColumnTypesT]):
    """Shared fields and validation for authored and resolved inference config."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    project_root: str
    preprocessing_data_path: Optional[str] = None
    metadata_config_path: _PathT = Field(default=None)
    model_path: Union[str, list[str]]
    model_type: str
    training_objective: str
    data_path: _PathT = Field(default=None)
    training_config_path: Optional[str] = Field(default="configs/train.yaml")
    read_format: str = Field(default="parquet")
    write_format: str = Field(default="csv")

    input_columns: _InputColumnsT
    target_columns: list[str]
    column_data_types: _ColumnTypesT = Field(default=None)
    target_column_types: _ColumnTypesT = Field(default=None)

    enforce_deterministic_inference: bool = Field(default=False)
    output_probabilities: bool = Field(default=False)
    map_to_id: bool = Field(default=True)
    seed: int = 1010
    device: str
    context_length: int = Field(gt=0)
    target_offset: int = Field(default=1, ge=0)
    model_window_stride: Optional[int] = Field(default=None, gt=0)
    prediction_length: Optional[int] = None
    inference_batch_size: int

    sample_from_distribution_columns: Optional[list[str]] = Field(default=None)
    infer_with_dropout: bool = Field(default=False)
    autoregression: bool = Field(default=False)
    autoregression_total_steps: Optional[int] = Field(default=None)

    @field_validator("input_columns", mode="before")
    @classmethod
    def normalize_single_input_column(cls, value):
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("training_objective")
    @classmethod
    def validate_authored_training_objective(cls, value: str) -> str:
        if value not in ALLOWED_OBJECTIVE_NAMES:
            raise ValueError(
                f"Only {OBJECTIVE_NAME_MESSAGE} are allowed, found {value}"
            )
        return value

    @field_validator("model_type")
    @classmethod
    def validate_authored_model_type(cls, value: str) -> str:
        if value not in {"embedding", "generative"}:
            raise ValueError("model_type must be either embedding or generative")
        return value

    @model_validator(mode="after")
    def validate_authored_paths(self):
        if self.metadata_config_path is None and self.preprocessing_data_path is None:
            raise ValueError(
                "metadata_config_path is required when preprocessing_data_path "
                "is not provided"
            )
        ModelWindowView(
            context_length=self.context_length,
            objective=self.training_objective,
            target_offset=target_offset_for_objective(
                self.training_objective, self.target_offset
            ),
        )
        return self


class InferenceConfig(
    _InferenceConfigBase[Optional[str], Optional[list[str]], Optional[dict[str, str]]]
):
    """User-authored configuration for one inference run."""


class ResolvedInferenceConfig(_InferenceConfigBase[str, list[str], dict[str, str]]):
    """Internal inference config after dataset metadata has been resolved."""

    metadata_config_path: str
    data_path: str
    input_columns: list[str]
    column_data_types: dict[str, str]
    target_column_types: dict[str, str]
    categorical_columns: list[str]
    real_columns: list[str]
    storage_layout: StoredWindowLayout
    window_view: ModelWindowView
    dataset_metadata: Optional[DatasetMetadata] = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def derive_optional_config_values(cls, values):
        if not isinstance(values, dict):
            return values
        values = dict(values)
        window_view = values.get("window_view")
        if "context_length" not in values and isinstance(window_view, dict):
            values["context_length"] = window_view.get("context_length")
        elif "context_length" not in values and isinstance(
            window_view, ModelWindowView
        ):
            values["context_length"] = window_view.context_length
        if "target_offset" not in values and isinstance(window_view, dict):
            values["target_offset"] = window_view.get("target_offset", 1)
        elif "target_offset" not in values and isinstance(window_view, ModelWindowView):
            values["target_offset"] = window_view.target_offset
        preprocessing_data_path = values.get("preprocessing_data_path")
        if values.get("metadata_config_path") is None and preprocessing_data_path:
            values["metadata_config_path"] = (
                metadata_config_path_from_preprocessing_data_path(
                    preprocessing_data_path
                )
            )
        if (
            values.get("target_column_types") is None
            and values.get("column_data_types") is not None
        ):
            values["target_column_types"] = derive_target_column_types(
                values.get("target_columns", []),
                values["column_data_types"],
            )
        return values

    @model_validator(mode="after")
    def validate_required_paths(self):
        if self.metadata_config_path is None:
            raise ValueError(
                "metadata_config_path is required when preprocessing_data_path "
                "is not provided"
            )
        if self.data_path is None:
            raise ValueError(
                "data_path must be provided or resolved from preprocessing metadata"
            )
        return self

    @model_validator(mode="after")
    def normalize_prediction_length(self):
        if self.window_view.objective != self.training_objective:
            raise ValueError(
                "window_view objective must match training_objective "
                f"({self.window_view.objective} != {self.training_objective})."
            )
        objective_class = get_objective_class(self.training_objective)
        if self.prediction_length is None:
            self.prediction_length = objective_class.default_prediction_length(
                self.window_view.context_length
            )
        objective_class.validate_prediction_length(
            self.prediction_length,
            self.window_view.context_length,
            usage="inference",
        )
        if objective_class.forward_looking:
            resolve_window_view(self.storage_layout, self.window_view)
        return self

    @field_validator("training_objective")
    @classmethod
    def validate_training_objective(cls, v):
        if v not in ALLOWED_OBJECTIVE_NAMES:
            raise ValueError(f"Only {OBJECTIVE_NAME_MESSAGE} are allowed, found {v}")
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

    @model_validator(mode="after")
    def validate_training_config_path(self):
        model_paths = (
            self.model_path if isinstance(self.model_path, list) else [self.model_path]
        )
        if not any(path.lower().endswith(".pt") for path in model_paths):
            return self

        if self.training_config_path is None:
            raise ValueError("training_config_path is required for PyTorch models")

        if not os.path.exists(self.training_config_path):
            raise ValueError(f"{self.training_config_path} does not exist")
        return self

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
            and not get_objective_class(
                info.data.get("training_objective")
            ).forward_looking
        ):
            raise ValueError(
                "Autoregressive inference is not possible with BERT-style models."
            )

        return v

    @field_validator("data_path")
    @classmethod
    def validate_data_path(
        cls, v: Optional[str], info: ValidationInfo
    ) -> Optional[str]:
        if v is None:
            return v
        v2 = normalize_path(v, info.data.get("project_root"))
        if not os.path.exists(v2):
            raise ValueError(f"{v2} does not exist")
        return v

    @field_validator("read_format")
    @classmethod
    def validate_read_format(cls, v: str) -> str:
        if v not in ["csv", "parquet", "pt"]:
            raise ValueError(
                "Currently only 'csv', 'parquet' and 'pt' are supported for "
                "inference input"
            )
        return v

    @field_validator("write_format")
    @classmethod
    def validate_write_format(cls, v: str) -> str:
        if v not in ["csv", "parquet"]:
            raise ValueError(
                "Currently only 'csv' and 'parquet' are supported for "
                "inference output"
            )
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

    @field_validator("column_data_types")
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
                "column_data_types must include every input column. "
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
        column_ordered = list(self.column_data_types.keys())
        columns_ordered_filtered = [
            c for c in column_ordered if c in self.target_columns
        ]
        if not (columns_ordered_filtered == self.target_columns):
            raise ValueError(f"{columns_ordered_filtered} != {self.target_columns}")


# Compatibility name retained for runtime code and external integrations.
InfererModel = ResolvedInferenceConfig
