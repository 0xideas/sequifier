"""Authored and resolved model interface schemas shared by execution and training."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import ConfigDict, Field, field_validator, model_validator

from sequifier.config.components import (
    BackboneComponentConfig,
    CategoricalHashingConfig,
    DecoderComponentConfig,
    FeatureLayoutRegistryModel,
    ImageEncoderConfig,
    IngestionComponentConfig,
)
from sequifier.config.identifiers import identifier as _identifier
from sequifier.config.image_config import image_views
from sequifier.config.metadata import (
    DatasetMetadata,
    FeatureMetadata,
    RawInputMetadata,
    TensorPayloadKind,
)
from sequifier.config.naming import ConfigModel
from sequifier.config.policy_config import PolicyBinding
from sequifier.config.prediction_config import PredictionDistributionConfig
from sequifier.helpers import ModelWindowView, StoredWindowLayout
from sequifier.special_tokens import SPECIAL_TOKEN_NAMES
from sequifier.typechecking import beartype


@beartype
def _unique_columns(value: list[str], usage: str) -> list[str]:
    if len(value) != len(set(value)):
        raise ValueError(f"{usage} cannot contain duplicate columns.")
    return value


class StaticConditioningConfig(ConfigModel):
    columns: list[str] = Field(min_length=1)
    total_embedding_dim: int | None = Field(default=None, gt=0)
    feature_embedding_dims: dict[str, int] | None = None

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, columns: list[str]) -> list[str]:
        return _unique_columns(columns, "static_conditioning.columns")

    @model_validator(mode="after")
    def validate_embedding_dims(self):
        dims = self.feature_embedding_dims
        if dims is None:
            return self
        if set(dims) != set(self.columns):
            raise ValueError(
                "static_conditioning.feature_embedding_dims must name "
                "exactly the configured static columns"
            )
        if any(dim <= 0 for dim in dims.values()):
            raise ValueError("Static feature embedding dimensions must be positive")
        if (
            self.total_embedding_dim is not None
            and sum(dims.values()) != self.total_embedding_dim
        ):
            raise ValueError(
                "Static feature embedding dimensions must sum to total_embedding_dim"
            )
        return self


class ModelInterfaceSpecModel(ConfigModel):
    """Architecture and selected-column contract for one named model route."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    input_columns: list[str] = Field(default_factory=list)
    prediction_distributions: dict[str, PredictionDistributionConfig] = Field(
        default_factory=dict
    )
    static_conditioning: StaticConditioningConfig | None = None
    preconditioning_image_encoder: ImageEncoderConfig | None = None
    target_columns: list[str] = Field(..., min_length=1)
    categorical_hashing: dict[str, CategoricalHashingConfig] = Field(
        default_factory=dict
    )
    categorical_decoder_special_tokens: dict[
        str, list[Literal["unknown", "other", "mask"]]
    ] = Field(default_factory=dict)
    feature_layout: Optional[FeatureLayoutRegistryModel] = None
    ingestion: IngestionComponentConfig
    decoder: DecoderComponentConfig

    @field_validator("input_columns", "target_columns")
    @classmethod
    @beartype
    def validate_columns(cls, value: list[str], info):
        if any(
            column.startswith("__image:")
            or column.startswith("__image_mask:")
            or column == "__preconditioning_image"
            for column in value
        ):
            raise ValueError("Image tensor key prefixes are reserved")
        return _unique_columns(value, info.field_name)

    @field_validator("categorical_decoder_special_tokens")
    @classmethod
    @beartype
    def validate_decoder_tokens(cls, value):
        if any(len(tokens) != len(set(tokens)) for tokens in value.values()):
            raise ValueError(
                "categorical_decoder_special_tokens cannot contain duplicate tokens"
            )
        return {
            column: [name for name in SPECIAL_TOKEN_NAMES if name in tokens]
            for column, tokens in value.items()
        }

    @model_validator(mode="after")
    @beartype
    def validate_interface_contract(self):
        if self.prediction_distributions.keys() - set(self.target_columns):
            raise ValueError("prediction_distributions must name model targets")
        input_columns = set(self.input_columns)
        if image_targets := set(self.target_columns) & image_views(self.ingestion):
            raise ValueError(
                f"Image views cannot be target columns: {sorted(image_targets)}"
            )
        if self.static_conditioning is not None:
            static_columns = set(self.static_conditioning.columns)
            if any(
                column.startswith("__image:")
                or column.startswith("__image_mask:")
                or column == "__preconditioning_image"
                for column in static_columns
            ):
                raise ValueError("Image tensor key prefixes are reserved")
            if static_columns & (input_columns | set(self.target_columns)):
                raise ValueError(
                    "Static conditioning columns must be separate from temporal "
                    "inputs and targets"
                )
        auxiliary_columns = set(self.ingestion.auxiliary_input_columns)
        if missing := auxiliary_columns - input_columns:
            raise ValueError(
                "ingestion.auxiliary_input_columns references unknown input "
                f"columns: {sorted(missing)}"
            )
        if self.feature_layout is not None:

            def image_cell_names(ingestion):
                if ingestion.type == "structured":
                    return set(ingestion.image_cells)
                if ingestion.type == "composite":
                    return set().union(
                        *(
                            image_cell_names(child)
                            for child in ingestion.branches.values()
                        )
                    )
                return set()

            image_cells = image_cell_names(self.ingestion)
            if image_cells & (input_columns | set(self.target_columns)):
                raise ValueError(
                    "Structured image cell names must differ from tabular columns"
                )
            for layout_name, layout in self.feature_layout.items():
                if missing := set(layout.columns) - input_columns - image_cells:
                    raise ValueError(
                        f"feature_layout {layout_name!r} references unknown "
                        f"columns outside input_columns: {sorted(missing)}"
                    )
        branches = (
            self.decoder.branches.items()
            if self.decoder.type == "composite"
            else (("default", self.decoder),)
        )
        for branch_name, branch in branches:
            if branch.type != "autoregressive_transformer":
                continue
            nonfinal_hashed_targets = [
                target
                for target in branch.target_columns[:-1]
                if target in self.categorical_hashing
            ]
            if nonfinal_hashed_targets:
                raise ValueError(
                    "Hashed targets must be last in their autoregressive_transformer "
                    f"decoder branch; branch {branch_name!r} has non-final hashed "
                    f"targets {nonfinal_hashed_targets!r}."
                )
        return self


class ModelSpecModel(ConfigModel):
    """One shared backbone and one or more named interfaces."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    backbone: BackboneComponentConfig
    interfaces: dict[str, ModelInterfaceSpecModel] = Field(..., min_length=1)

    @field_validator("interfaces")
    @classmethod
    @beartype
    def validate_interface_names(cls, value):
        for name in value:
            _identifier(name, "Model interface name")
        return value


class ResolvedInterfaceStorage(RawInputMetadata):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    tensor_payload_kind: TensorPayloadKind
    tensor_payload_version: int = Field(strict=True)
    split_paths: list[str] = Field(default_factory=list)
    prediction_aligned_splits: list[int] = Field(default_factory=list)
    prediction_alignment_steps: int = 1
    window_lengths: Optional[list[int]] = None
    prediction_length: Optional[int] = None
    target_offset: Optional[int] = None
    static_features_path: str | None = None
    storage_layout: StoredWindowLayout

    @classmethod
    def from_metadata(
        cls, metadata: DatasetMetadata, layout: StoredWindowLayout | None = None
    ) -> ResolvedInterfaceStorage:
        values = {name: getattr(metadata, name) for name in cls.model_fields}
        values["storage_layout"] = layout or metadata.storage_layout
        return cls(**values)


class ResolvedModelInterface(ModelInterfaceSpecModel, FeatureMetadata):
    column_data_types: dict[str, str] = Field(...)
    n_classes: dict[str, int] = Field(...)
    id_maps: dict[str, dict[str | int, int]] = Field(...)
    special_token_ids: dict[str, int] = Field(...)
    policy: PolicyBinding | None = None
    name: str
    static_categorical_columns: list[str] = Field(default_factory=list)
    static_real_columns: list[str] = Field(default_factory=list)
    categorical_hash_contracts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    target_column_types: dict[str, str]
    categorical_columns: list[str]
    real_columns: list[str]
    target_decoder_ids: dict[str, list[int]]
    target_n_classes: dict[str, int]
    target_global_to_decoder: dict[str, list[int]]
    window_view: ModelWindowView

    @field_validator("name")
    @classmethod
    def validate_name(cls, name):
        return _identifier(name, "Model interface name")

    def preprocessing_metadata(
        self, storage: ResolvedInterfaceStorage | None
    ) -> DatasetMetadata:
        if storage is None:
            raise ValueError(f"{self.name}: stored-window metadata is required")
        layout = storage.storage_layout
        return DatasetMetadata.model_validate(
            {
                **self.feature_values(),
                **storage.model_dump(exclude={"storage_layout"}),
                "static_columns": self.static_conditioning.columns
                if self.static_conditioning
                else [],
                "window_length": storage.window_lengths[0]
                if storage.window_lengths
                else layout.window_length,
                "max_target_offset": layout.max_target_offset,
                "stored_window_layout_version": layout.version,
            },
            context={"projected_depth_layouts": True},
        )

    @model_validator(mode="after")
    def validate_prediction_targets(self):
        for name in self.prediction_distributions:
            if (
                name not in self.target_columns
                or self.target_column_types.get(name) != "real"
            ):
                raise ValueError(
                    f"{name}: prediction_distributions requires a real model target"
                )
        return self
