import os
from typing import Any, Generic, Optional, TypeVar, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
from sequifier.typechecking import beartype


@beartype
def _comparable_value(field: str, value: Any) -> Any:
    if field == "column_data_types" and isinstance(value, dict):
        return {
            column: canonicalize_polars_dtype_name(dtype)
            for column, dtype in value.items()
        }
    return value


@beartype
def _merge_loaded_execution_values(
    values: dict[str, Any], loaded: dict[str, Any], source: str
) -> None:
    """Fill omitted model fields and reject authored compatibility mismatches."""

    for field, loaded_value in loaded.items():
        configured_value = values.get(field)
        if configured_value is not None and _comparable_value(
            field, configured_value
        ) != _comparable_value(field, loaded_value):
            raise ValueError(
                f"Inference config field {field!r} does not match {source}: "
                f"configured {configured_value!r}, loaded {loaded_value!r}."
            )
        values[field] = loaded_value


@beartype
def _execution_source(
    training: Any, dataset: Any
) -> tuple[dict[str, Any], DatasetMetadata]:
    interface = dataset.interface
    global_spec = training.global_training
    layout = interface.storage_layout
    return (
        {
            "model_interface": dataset.model_interface,
            "input_columns": list(interface.input_columns),
            "target_columns": list(interface.target_columns),
            "column_data_types": dict(interface.column_data_types),
            "target_column_types": dict(interface.target_column_types),
            "training_objective": global_spec.training_objective,
            "context_length": global_spec.context_length,
            "target_offset": global_spec.target_offset,
            "prediction_length": interface.decoder.prediction_length,
        },
        DatasetMetadata(
            column_data_types=dict(interface.column_data_types),
            n_classes=dict(interface.n_classes),
            id_maps=dict(interface.id_maps),
            special_token_ids=dict(interface.special_token_ids),
            selected_columns_statistics=dict(interface.selected_columns_statistics),
            normalize_real_columns=interface.normalize_real_columns,
            window_length=layout.window_length,
            max_target_offset=layout.max_target_offset,
            stored_window_layout_version=layout.version,
        ),
    )


@beartype
def _assert_metadata_matches_source(
    loaded_metadata: DatasetMetadata,
    selected_metadata: DatasetMetadata,
    source: str,
    layout_is_authoritative: bool,
) -> None:
    for field in (
        "column_data_types",
        "n_classes",
        "id_maps",
        "selected_columns_statistics",
    ):
        expected = getattr(loaded_metadata, field)
        found = getattr(selected_metadata, field)
        found = {key: found.get(key) for key in expected}
        if _comparable_value(field, found) != _comparable_value(field, expected):
            raise ValueError(
                f"Inference metadata field {field!r} does not match {source}."
            )
    for field in ("special_token_ids", "normalize_real_columns"):
        expected = getattr(loaded_metadata, field)
        found = getattr(selected_metadata, field)
        if found != expected:
            raise ValueError(
                f"Inference metadata field {field!r} does not match {source}: "
                f"configured {found!r}, loaded {expected!r}."
            )
    if (
        layout_is_authoritative
        and selected_metadata.storage_layout != loaded_metadata.storage_layout
    ):
        raise ValueError(
            "Inference metadata storage_layout does not match "
            f"{source}: configured {selected_metadata.storage_layout!r}, loaded "
            f"{loaded_metadata.storage_layout!r}."
        )


@beartype
def load_inferer_config(
    config_path: str, args_config: dict, skip_metadata: bool
) -> "ResolvedInferenceConfig":
    """Compose inference YAML, load model defaults, and resolve metadata."""

    config_values = load_composed_yaml_config(config_path)

    cli_values = {
        key: value for key, value in args_config.items() if key != "skip_metadata"
    }
    authored_values = merge_config_fragments((config_values, cli_values))
    authored_values, extracted_metadata_values = extract_inline_metadata(
        authored_values
    )
    inline_metadata_values = extracted_metadata_values if skip_metadata else None

    project_root = authored_values["project_root"]
    selected_metadata = None
    selected_metadata_path = None
    loaded_metadata_sources: list[tuple[DatasetMetadata, str, bool]] = []

    dataset_name = authored_values.get("dataset")
    model_paths = authored_values.get("model_path")
    if isinstance(model_paths, str):
        model_paths = [model_paths]
    training_config_path = authored_values.get("training_config_path")
    route_selected = (
        dataset_name is not None or authored_values.get("model_interface") is not None
    )
    if training_config_path is not None and (
        route_selected
        or (
            isinstance(model_paths, list)
            and ("model_path" not in cli_values or "training_config_path" in cli_values)
            and any(
                isinstance(path, str) and path.lower().endswith(".pt")
                for path in model_paths
            )
        )
    ):
        from sequifier.config.train_config import load_train_config

        training = load_train_config(training_config_path, {}, False)
        datasets = training.dataset_training
        if dataset_name is not None:
            if dataset_name not in datasets:
                raise ValueError(f"Unknown inference dataset {dataset_name!r}")
            dataset = datasets[dataset_name]
        elif authored_values.get("model_interface") is not None:
            interface_name = authored_values["model_interface"]
            dataset = next(
                (
                    candidate
                    for candidate in datasets.values()
                    if candidate.model_interface == interface_name
                ),
                None,
            )
            if dataset is None:
                raise ValueError(f"Unknown model interface {interface_name!r}")
        elif len(datasets) == 1:
            dataset = next(iter(datasets.values()))
        else:
            raise ValueError(
                "A dataset or model_interface selection is required for the "
                "training config"
            )
        loaded_values, training_metadata = _execution_source(training, dataset)
        source = f"training config dataset {dataset.name!r}"
        _merge_loaded_execution_values(authored_values, loaded_values, source)
        loaded_metadata_sources.append((training_metadata, source, True))

        if dataset_name is not None:
            part_name = authored_values.get("part")
            if part_name is None:
                if len(dataset.parts) != 1:
                    raise ValueError(
                        f"Dataset {dataset_name!r} has multiple parts; select part"
                    )
                part_name = next(iter(dataset.parts))
                authored_values["part"] = part_name
            if part_name not in dataset.parts:
                raise ValueError(f"Unknown inference part {dataset_name}.{part_name}")
            part = dataset.parts[part_name]
            selected_metadata = part.metadata
            selected_metadata_path = part.metadata_config_path

    import torch

    from sequifier.artifacts.model_config import resolved_config_from_model_config

    if isinstance(model_paths, list):
        for model_path in model_paths:
            if not isinstance(model_path, str) or not model_path.lower().endswith(
                ".pt"
            ):
                continue
            payload = torch.load(
                normalize_path(model_path, project_root),
                map_location="cpu",
                weights_only=False,
            )
            model_config = payload.get("model_config")
            if model_config is None:
                continue
            if (
                "model_path" in cli_values
                and "training_config_path" not in cli_values
                and not route_selected
            ):
                authored_values["training_config_path"] = None
            training, interface_name = resolved_config_from_model_config(
                model_config,
                device=str(authored_values.get("device", "cpu")),
                interface_name=authored_values.get("model_interface"),
            )
            dataset = training.dataset_training[interface_name]
            loaded_values, artifact_metadata = _execution_source(training, dataset)
            source = f"PT artifact {model_path!r}"
            _merge_loaded_execution_values(authored_values, loaded_values, source)
            loaded_metadata_sources.append(
                (
                    artifact_metadata,
                    source,
                    "storage_layout" in model_config["interfaces"][interface_name],
                )
            )

    config = try_catch_excess_keys(config_path, InferenceConfig, authored_values)
    metadata_path = _effective_metadata_config_path(config)
    if skip_metadata and inline_metadata_values is not None:
        metadata = DatasetMetadata.model_validate(inline_metadata_values)
    elif not skip_metadata and metadata_path is not None:
        metadata = load_dataset_metadata(
            normalize_path(metadata_path, config.project_root)
        )
    elif selected_metadata is not None:
        metadata = selected_metadata
    elif loaded_metadata_sources:
        metadata = loaded_metadata_sources[0][0]
    elif skip_metadata:
        raise ValueError(
            "skip_metadata requires inline storage_layout and column values, a "
            "selected training dataset, or a self-describing PT artifact."
        )
    else:
        raise ValueError(
            f"Inference config '{config_path}' must define metadata_config_path "
            "or preprocessing_data_path unless the PT artifact contains model "
            "metadata."
        )

    if metadata_path is None and selected_metadata_path is not None:
        config.metadata_config_path = selected_metadata_path
    for loaded_metadata, source, layout_is_authoritative in loaded_metadata_sources:
        _assert_metadata_matches_source(
            loaded_metadata,
            metadata,
            source,
            layout_is_authoritative,
        )

    return resolve_inference_config(config, metadata)


@beartype
def _effective_metadata_config_path(config: "InferenceConfig") -> Optional[str]:
    if config.metadata_config_path:
        return config.metadata_config_path
    if config.preprocessing_data_path:
        return metadata_config_path_from_preprocessing_data_path(
            config.preprocessing_data_path
        )
    return None


@beartype
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


_DataPathT = TypeVar("_DataPathT")
_InputColumnsT = TypeVar("_InputColumnsT")
_ColumnTypesT = TypeVar("_ColumnTypesT")


class _InferenceConfigBase(
    BaseModel, Generic[_DataPathT, _InputColumnsT, _ColumnTypesT]
):
    """Shared fields and validation for authored and resolved inference config."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    project_root: str
    preprocessing_data_path: Optional[str] = None
    metadata_config_path: Optional[str] = None
    model_path: Union[str, list[str]]
    model_type: str
    training_objective: str
    data_path: _DataPathT
    training_config_path: Optional[str] = None
    dataset: Optional[str] = None
    part: Optional[str] = None
    model_interface: Optional[str] = None
    read_format: str = Field(default="parquet")
    write_format: str = Field(default="csv")

    input_columns: _InputColumnsT
    target_columns: list[str]
    column_data_types: _ColumnTypesT
    target_column_types: _ColumnTypesT

    deterministic: bool = Field(default=False)
    output_probabilities: bool = Field(default=False)
    decode_categories: bool = Field(default=True)
    seed: int = 1010
    device: str
    context_length: int = Field(gt=0)
    target_offset: int = Field(default=1, ge=0)
    window_stride: Optional[int] = Field(default=None, gt=0)
    prediction_length: Optional[int] = None
    inference_batch_size: int = Field(default=1, gt=0)

    sample_from_distribution_columns: Optional[list[str]] = Field(default=None)
    infer_with_dropout: bool = Field(default=False)
    autoregressive: bool = Field(default=False)
    generation_steps: Optional[int] = Field(default=None)

    @model_validator(mode="after")
    @beartype
    def validate_route_selection(self):
        for usage, value in (
            ("dataset", self.dataset),
            ("part", self.part),
            ("model_interface", self.model_interface),
        ):
            if value is not None and ("." in value or not value.isidentifier()):
                raise ValueError(f"{usage} must be a valid identifier without '.'")
        if self.part is not None and self.dataset is None:
            raise ValueError("part selection requires dataset selection")
        return self

    @field_validator("input_columns", mode="before")
    @classmethod
    @beartype
    def normalize_single_input_column(cls, value):
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("training_objective")
    @classmethod
    @beartype
    def validate_authored_training_objective(cls, value: str) -> str:
        if value not in ALLOWED_OBJECTIVE_NAMES:
            raise ValueError(
                f"Only {OBJECTIVE_NAME_MESSAGE} are allowed, found {value}"
            )
        return value

    @field_validator("model_type")
    @classmethod
    @beartype
    def validate_authored_model_type(cls, value: str) -> str:
        if value not in {"embedding", "generative"}:
            raise ValueError("model_type must be either embedding or generative")
        return value

    @model_validator(mode="after")
    @beartype
    def validate_authored_paths(self):
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

    data_path: Optional[str] = None
    column_data_types: Optional[dict[str, str]] = None
    target_column_types: Optional[dict[str, str]] = None


class ResolvedInferenceConfig(_InferenceConfigBase[str, list[str], dict[str, str]]):
    """Internal inference config after dataset metadata has been resolved."""

    categorical_columns: list[str]
    real_columns: list[str]
    storage_layout: StoredWindowLayout
    window_view: ModelWindowView
    dataset_metadata: Optional[DatasetMetadata] = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    @beartype
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
    @beartype
    def validate_required_paths(self):
        if self.data_path is None:
            raise ValueError(
                "data_path must be provided or resolved from preprocessing metadata"
            )
        return self

    @model_validator(mode="after")
    @beartype
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
    @beartype
    def validate_training_objective(cls, v):
        if v not in ALLOWED_OBJECTIVE_NAMES:
            raise ValueError(f"Only {OBJECTIVE_NAME_MESSAGE} are allowed, found {v}")
        return v

    @field_validator("model_type")
    @classmethod
    @beartype
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
    @beartype
    def validate_output_probabilities(cls, v: bool, info: Any) -> bool:
        if v and info.data.get("model_type") == "embedding":
            raise ValueError(
                "For embedding models, 'output_probabilities' must be set to false"
            )
        return v

    @model_validator(mode="after")
    @beartype
    def validate_training_config_path(self):
        model_paths = (
            self.model_path if isinstance(self.model_path, list) else [self.model_path]
        )
        if not any(path.lower().endswith(".pt") for path in model_paths):
            return self

        if self.training_config_path is not None and not os.path.exists(
            self.training_config_path
        ):
            raise ValueError(f"{self.training_config_path} does not exist")
        return self

    @field_validator("generation_steps")
    @classmethod
    @beartype
    def validate_generation_steps(cls, v: Optional[int], info: Any) -> Optional[int]:
        if v is None and info.data.get("autoregressive") is True:
            raise ValueError(
                "If autoregressive==True, 'generation_steps' needs to be set to an integer value."
            )
        if v is not None and v < 1:
            raise ValueError("generation_steps must by >= 1.")
        if v is not None and v > 1:
            if not info.data.get("autoregressive"):
                raise ValueError(
                    f"'generation_steps' can only be larger than 1 if 'autoregressive' is true: {info.data.get('autoregressive')}"
                )

            if not np.all(
                np.array(sorted(info.data.get("input_columns")))
                == np.array(sorted(info.data.get("target_columns")))
            ):
                raise ValueError(
                    "'generation_steps' can only be larger than 1 if 'input_columns' and 'target_columns' are identical"
                )

        return v

    @field_validator("autoregressive")
    @classmethod
    @beartype
    def validate_autoregressive(cls, v: bool, info: Any):
        if v and info.data.get("model_type") == "embedding":
            raise ValueError(
                "Autoregressive inference is not possible for embedding models"
            )
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
    @beartype
    def validate_data_path(cls, v: Optional[str], info: Any) -> Optional[str]:
        if v is None:
            return v
        v2 = normalize_path(v, info.data.get("project_root"))
        if not os.path.exists(v2):
            raise ValueError(f"{v2} does not exist")
        return v

    @field_validator("read_format")
    @classmethod
    @beartype
    def validate_read_format(cls, v: str) -> str:
        if v not in ["csv", "parquet", "pt"]:
            raise ValueError(
                "Currently only 'csv', 'parquet' and 'pt' are supported for "
                "inference input"
            )
        return v

    @field_validator("write_format")
    @classmethod
    @beartype
    def validate_write_format(cls, v: str) -> str:
        if v not in ["csv", "parquet"]:
            raise ValueError(
                "Currently only 'csv' and 'parquet' are supported for inference output"
            )
        return v

    @field_validator("target_column_types")
    @classmethod
    @beartype
    def validate_target_column_types(
        cls, v: Optional[dict], info: Any
    ) -> Optional[dict]:
        if v is None:
            return v
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
    @beartype
    def validate_column_types(cls, v: Optional[dict], info: Any) -> Optional[dict]:
        if v is None:
            return v
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

    @beartype
    def __init__(self, **data):
        super().__init__(**data)
        if self.column_data_types is None:
            return
        column_ordered = list(self.column_data_types.keys())
        columns_ordered_filtered = [
            c for c in column_ordered if c in self.target_columns
        ]
        if not (columns_ordered_filtered == self.target_columns):
            raise ValueError(f"{columns_ordered_filtered} != {self.target_columns}")


# Compatibility name retained for runtime code and external integrations.
InfererModel = ResolvedInferenceConfig
