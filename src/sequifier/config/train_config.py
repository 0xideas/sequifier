import copy
import math
import os
import warnings
from dataclasses import dataclass
from itertools import product
from typing import Annotated, Any, Generic, Literal, Optional, TypeAlias, TypeVar, Union

import torch
import torch_optimizer
from beartype import beartype
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StrictInt,
    StrictStr,
    field_serializer,
    field_validator,
    model_validator,
)

import sequifier
import sequifier.optimizers
from sequifier.config.composition import (
    load_composed_yaml_config,
    merge_config_fragments,
)
from sequifier.config.initialization_config import ModelInitializationConfig
from sequifier.config.metadata import (
    DatasetMetadata,
    extract_inline_metadata,
    load_dataset_metadata,
)
from sequifier.config.probabilities import ProbabilityDistribution
from sequifier.helpers import (
    ModelWindowView,
    StoredWindowLayout,
    derive_target_column_types,
    metadata_config_path_from_preprocessing_data_path,
    normalize_path,
    resolve_window_view,
    try_catch_excess_keys,
)
from sequifier.objectives import (
    ALLOWED_OBJECTIVE_NAMES,
    OBJECTIVE_NAME_MESSAGE,
    BERTObjective,
    NextOccurrenceObjective,
    get_objective_class,
    target_offset_for_objective,
)
from sequifier.special_tokens import (
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKEN_NAMES,
    resolve_categorical_decoder_ids,
    validate_special_token_ids,
)

AnyType = str | int | float
NextOccurrenceTargetValue: TypeAlias = StrictInt | StrictStr


class CartesianLayoutModel(BaseModel):
    """Reusable coordinate annotation for flat feature columns."""

    model_config = ConfigDict(extra="forbid")

    axes: dict[str, list[AnyType]] = Field(..., min_length=1)
    columns: dict[str, dict[str, AnyType]] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_cartesian(self):
        axis_names = list(self.axes)

        for axis, values in self.axes.items():
            if not values:
                raise ValueError(
                    f"Layout axis {axis!r} must contain at least one value"
                )
            if len(values) != len(set(values)):
                raise ValueError(
                    f"Layout axis {axis!r} cannot contain duplicate values"
                )

        coordinate_tuples = set()
        for column_name, coordinates in self.columns.items():
            if set(coordinates) != set(axis_names):
                raise ValueError(
                    f"Layout column {column_name!r} must define every axis"
                )

            coordinate_tuple = tuple(coordinates[axis] for axis in axis_names)
            if coordinate_tuple in coordinate_tuples:
                raise ValueError(
                    f"Duplicate cartesian coordinate tuple: {coordinate_tuple!r}"
                )
            coordinate_tuples.add(coordinate_tuple)

            for axis, value in coordinates.items():
                if value not in self.axes[axis]:
                    raise ValueError(
                        f"Layout column {column_name!r} has value {value!r} "
                        f"outside axis {axis!r}"
                    )

        expected_tuples = set(product(*(self.axes[axis] for axis in axis_names)))
        if coordinate_tuples != expected_tuples:
            raise ValueError("cartesian layouts must contain every coordinate")

        return self


class FeatureLayoutRegistryModel(RootModel[dict[str, CartesianLayoutModel]]):
    """Top-level registry of reusable feature layouts."""

    root: dict[str, CartesianLayoutModel] = Field(..., min_length=1)

    def items(self):
        return self.root.items()

    def __contains__(self, key: str) -> bool:
        return key in self.root

    def __getitem__(self, key: str) -> CartesianLayoutModel:
        return self.root[key]


class IngestionComponentBase(BaseModel):
    """Settings shared by every feature-ingestion component."""

    model_config = ConfigDict(extra="forbid")

    allow_shared_columns: bool = False
    allow_unused_input_columns: bool = False
    auxiliary_input_columns: list[str] = Field(default_factory=list)
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    initialization: ModelInitializationConfig = Field(
        default_factory=ModelInitializationConfig
    )

    @field_validator("auxiliary_input_columns")
    @classmethod
    def validate_auxiliary_input_columns(cls, value):
        _validate_column_list_unique(
            value, "model_spec.ingestion.auxiliary_input_columns"
        )
        return value


class DirectEmbedIngestionConfig(IngestionComponentBase):
    """Use the existing flat-column embedding path."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["direct_embed"] = "direct_embed"
    columns: Optional[list[str]] = Field(default=None, min_length=1)
    output_dim: int = Field(..., gt=0)
    feature_embedding_dims: Optional[dict[str, int]] = None

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, v):
        if v is not None:
            _validate_column_list_unique(v, "direct_embed ingestion columns")
        return v

    @field_validator("feature_embedding_dims")
    @classmethod
    def validate_feature_embedding_dims(cls, v):
        _validate_feature_embedding_dims(v, "direct_embed feature_embedding_dims")
        return v


class PassThroughIngestionConfig(IngestionComponentBase):
    """Pass real-valued columns through without per-feature encoders."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["pass_through"]
    columns: Optional[list[str]] = Field(default=None, min_length=1)
    output_dim: int = Field(..., gt=0)

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, v):
        if v is not None:
            _validate_column_list_unique(v, "pass_through ingestion columns")
        return v


class FeaturePoolIngestionConfig(IngestionComponentBase):
    """Encode columns as feature tokens before pooling to one time token."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["feature_pool"]
    columns: list[str] = Field(..., min_length=1)
    output_dim: int = Field(..., gt=0)

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, v):
        _validate_column_list_unique(v, "feature_pool ingestion columns")
        return v


class GroupedIngestionConfig(IngestionComponentBase):
    """Encode configured column groups and merge them within one branch."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["grouped"]
    output_dim: int = Field(..., gt=0)
    groups: dict[str, list[str]] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_groups(self):
        grouped_columns = []
        for group_name, columns in self.groups.items():
            _validate_module_dict_key(
                group_name, f"grouped ingestion group {group_name!r}"
            )
            if not columns:
                raise ValueError(
                    f"grouped ingestion group {group_name!r} must contain columns"
                )
            _validate_column_list_unique(
                columns, f"grouped ingestion group {group_name!r}"
            )
            grouped_columns.extend(columns)

        _validate_column_list_unique(grouped_columns, "grouped ingestion columns")
        return self


class SiameseIngestionConfig(IngestionComponentBase):
    """Apply one shared scalar encoder across the branch columns."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["siamese"]
    columns: list[str] = Field(..., min_length=1)
    output_dim: int = Field(..., gt=0)

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, v):
        _validate_column_list_unique(v, "siamese ingestion columns")
        return v


class TemporalConvIngestionConfig(IngestionComponentBase):
    """Encode columns, then apply Conv1D over the time axis."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["temporal_conv"]
    columns: list[str] = Field(..., min_length=1)
    output_dim: int = Field(..., gt=0)
    base_ingestion: Literal["direct_embed", "pass_through"] = "direct_embed"
    feature_embedding_dims: Optional[dict[str, int]] = None
    kernel_size: int = Field(3, gt=0)
    dilation: int | list[int] = 1
    num_layers: int = Field(1, gt=0)
    causal: bool = True
    activation_fn: Literal["relu", "gelu", "silu"] = "gelu"
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    post_conv_norm: Literal["layer_norm", "rmsnorm", "none"] = "layer_norm"
    orientation: Literal["within_column", "within_item_position"] = (
        "within_item_position"
    )

    @model_validator(mode="before")
    @classmethod
    def default_num_layers_from_dilation_schedule(cls, values):
        if not isinstance(values, dict):
            return values
        dilation = values.get("dilation")
        if isinstance(dilation, list) and "num_layers" not in values:
            values = dict(values)
            values["num_layers"] = len(dilation)
        return values

    @model_validator(mode="after")
    def validate_temporal_conv(self):
        _validate_column_list_unique(self.columns, "temporal_conv ingestion columns")
        _validate_feature_embedding_dims(
            self.feature_embedding_dims, "temporal_conv feature_embedding_dims"
        )
        if self.base_ingestion == "pass_through" and self.feature_embedding_dims:
            raise ValueError(
                "temporal_conv feature_embedding_dims is only valid when "
                "base_ingestion is 'direct_embed'"
            )
        if isinstance(self.dilation, list):
            invalid_dilation_values = [d for d in self.dilation if d <= 0]
            if invalid_dilation_values:
                raise ValueError(
                    "temporal_conv dilation schedule must contain positive "
                    f"integers: {invalid_dilation_values}"
                )
            if len(self.dilation) != self.num_layers:
                raise ValueError(
                    "temporal_conv dilation schedule length must equal "
                    f"num_layers: {len(self.dilation)} != {self.num_layers}"
                )
        elif self.dilation <= 0:
            raise ValueError("temporal_conv dilation must be positive")
        if not self.causal and self.kernel_size % 2 == 0:
            raise ValueError(
                "temporal_conv kernel_size must be odd when causal is false"
            )
        return self

    @property
    def dilation_schedule(self) -> list[int]:
        if isinstance(self.dilation, list):
            return self.dilation
        return [self.dilation] * self.num_layers


class AxisProjectionBlockModel(BaseModel):
    """Flatten configured cartesian axes and project them with a linear layer."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["axis_projection"]
    axes: list[str] = Field(..., min_length=1)
    output_dim: int = Field(..., gt=0)
    unshared_axes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_axes_unique(self):
        _validate_axis_list_unique(self.axes, "axes")
        _validate_axis_list_unique(self.unshared_axes, "unshared_axes")
        return self


class AxisConvBlockModel(BaseModel):
    """Sweep a native 1D/2D/3D convolution over configured cartesian axes."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["axis_conv"]
    axes: list[str] = Field(..., min_length=1, max_length=3)
    output_dim: int = Field(..., gt=0)
    kernel_size: int = Field(3, gt=0)
    unshared_axes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_axes_unique(self):
        _validate_axis_list_unique(self.axes, "axes")
        _validate_axis_list_unique(self.unshared_axes, "unshared_axes")
        if self.kernel_size % 2 == 0:
            raise ValueError("axis_conv kernel_size must be odd to preserve axis sizes")
        return self


class AxisAttentionBlockModel(BaseModel):
    """Apply self-attention over one or more configured cartesian axes."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["axis_attention"]
    axes: list[str] = Field(..., min_length=1)
    output_dim: int = Field(..., gt=0)
    n_head: int = Field(1, gt=0)
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    unshared_axes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_axes_unique(self):
        _validate_axis_list_unique(self.axes, "axes")
        _validate_axis_list_unique(self.unshared_axes, "unshared_axes")
        if self.output_dim % self.n_head != 0:
            raise ValueError("axis_attention output_dim must be divisible by n_head")
        return self


class AxisPoolBlockModel(BaseModel):
    """Reduce configured cartesian axes without changing the channel dimension."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["axis_pool"]
    axes: list[str] = Field(..., min_length=1)
    mode: Literal["mean", "sum", "max"] = "mean"

    @model_validator(mode="after")
    def validate_axes_unique(self):
        _validate_axis_list_unique(self.axes, "axes")
        return self


StructuredProcessingBlock = Annotated[
    Union[
        AxisProjectionBlockModel,
        AxisConvBlockModel,
        AxisAttentionBlockModel,
        AxisPoolBlockModel,
    ],
    Field(discriminator="type"),
]


def _validate_axis_list_unique(axes: list[str], field_name: str) -> None:
    if len(axes) != len(set(axes)):
        raise ValueError(f"{field_name} cannot contain duplicate axes")


def _validate_column_list_unique(columns: list[str], field_name: str) -> None:
    if len(columns) != len(set(columns)):
        raise ValueError(f"{field_name} cannot contain duplicate columns")


def _validate_module_dict_key(key: str, usage: str) -> None:
    if key == "":
        raise ValueError(f"{usage} cannot be empty")
    if "." in key:
        raise ValueError(f"{usage} cannot contain '.'")


def _validate_feature_embedding_dims(
    feature_embedding_dims: Optional[dict[str, int]], field_name: str
) -> None:
    if feature_embedding_dims is None:
        return
    invalid_dims = {
        column: dim for column, dim in feature_embedding_dims.items() if dim <= 0
    }
    if invalid_dims:
        raise ValueError(f"{field_name} values must be positive: {invalid_dims}")


class AxisEmbeddingModel(BaseModel):
    """Optional positional encoding for cartesian layout axes."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["none", "learned", "rope"] = "none"
    axes: list[str] = Field(default_factory=list)
    rope_theta: float = Field(10000.0, gt=0.0)

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, v):
        if v is None:
            return "none"
        if isinstance(v, str):
            return v.lower()
        return v

    @model_validator(mode="after")
    def validate_axes(self):
        _validate_axis_list_unique(self.axes, "axis_embeddings.axes")
        if self.type == "none" and self.axes:
            raise ValueError("axis_embeddings.axes must be empty when type is 'none'")
        if self.type != "none" and not self.axes:
            raise ValueError(
                "axis_embeddings.axes must contain at least one axis unless type is 'none'"
            )
        return self


class StructuredIngestionConfig(IngestionComponentBase):
    """Consume a top-level cartesian layout."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["structured"]
    layout: str
    output_dim: int = Field(..., gt=0)
    cell_dim: Optional[int] = Field(default=None, gt=0)
    axis_embeddings: AxisEmbeddingModel = Field(default_factory=AxisEmbeddingModel)
    processing_blocks: list[StructuredProcessingBlock] = Field(default_factory=list)

    @field_validator("axis_embeddings", mode="before")
    @classmethod
    def normalize_axis_embeddings(cls, v):
        if v is None:
            return {"type": "none", "axes": []}
        if isinstance(v, list):
            return {"type": "learned", "axes": v}
        return v


BranchIngestionConfig = Annotated[
    Union[
        DirectEmbedIngestionConfig,
        PassThroughIngestionConfig,
        FeaturePoolIngestionConfig,
        GroupedIngestionConfig,
        SiameseIngestionConfig,
        TemporalConvIngestionConfig,
        StructuredIngestionConfig,
    ],
    Field(discriminator="type"),
]


class IngestionMergeConfig(BaseModel):
    """How composite branch outputs are merged."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["concat", "sum", "gated", "attention"] = "concat"


IngestionSpecConfig = BranchIngestionConfig | dict[str, BranchIngestionConfig]


class CompositeIngestionConfig(IngestionComponentBase):
    """Combine independently configured ingestion branches."""

    type: Literal["composite"]
    branches: dict[str, BranchIngestionConfig] = Field(..., min_length=1)
    merge: IngestionMergeConfig = Field(default_factory=IngestionMergeConfig)

    @field_validator("branches")
    @classmethod
    def validate_branch_names(cls, branches):
        for branch_name in branches:
            _validate_module_dict_key(
                branch_name, f"Composite ingestion branch {branch_name!r}"
            )
        return branches


IngestionComponentConfig = Annotated[
    Union[
        DirectEmbedIngestionConfig,
        PassThroughIngestionConfig,
        FeaturePoolIngestionConfig,
        GroupedIngestionConfig,
        SiameseIngestionConfig,
        TemporalConvIngestionConfig,
        StructuredIngestionConfig,
        CompositeIngestionConfig,
    ],
    Field(discriminator="type"),
]


class LinearDecodingConfig(BaseModel):
    """Project each support window directly to target outputs."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["linear"] = "linear"
    target_columns: Optional[list[str]] = Field(default=None, min_length=1)

    @field_validator("target_columns")
    @classmethod
    def validate_target_columns(cls, v):
        if v is not None:
            _validate_column_list_unique(v, "linear decoding target_columns")
        return v


class MLPDecodingConfig(BaseModel):
    """Flatten support windows and decode targets with a shared MLP branch."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["mlp"]
    target_columns: Optional[list[str]] = Field(default=None, min_length=1)
    hidden_dims: list[int] = Field(..., min_length=1)
    activation_fn: Literal["relu", "gelu", "silu"] = "relu"
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    hidden_weight_l2: float = Field(
        0.0,
        ge=0.0,
        allow_inf_nan=False,
    )

    @field_validator("target_columns")
    @classmethod
    def validate_target_columns(cls, v):
        if v is not None:
            _validate_column_list_unique(v, "mlp decoding target_columns")
        return v

    @field_validator("hidden_dims")
    @classmethod
    def validate_hidden_dims(cls, v):
        invalid_dims = [dim for dim in v if dim <= 0]
        if invalid_dims:
            raise ValueError(
                f"mlp decoding hidden_dims must be positive: {invalid_dims}"
            )
        return v


BranchDecodingConfig = Annotated[
    Union[
        LinearDecodingConfig,
        MLPDecodingConfig,
    ],
    Field(discriminator="type"),
]


DecodingSpecConfig = BranchDecodingConfig | dict[str, BranchDecodingConfig]


class DecoderComponentBase(BaseModel):
    """Settings shared by every target-decoder component."""

    model_config = ConfigDict(extra="forbid")

    prediction_length: int = Field(..., gt=0)
    support: int = Field(1, gt=0)
    initialization: ModelInitializationConfig = Field(
        default_factory=ModelInitializationConfig
    )


class LinearDecoderComponentConfig(DecoderComponentBase, LinearDecodingConfig):
    pass


class MLPDecoderComponentConfig(DecoderComponentBase, MLPDecodingConfig):
    pass


class CompositeDecoderComponentConfig(DecoderComponentBase):
    type: Literal["composite"]
    branches: dict[str, BranchDecodingConfig] = Field(..., min_length=1)

    @field_validator("branches")
    @classmethod
    def validate_branch_names(cls, branches):
        for branch_name in branches:
            _validate_module_dict_key(
                branch_name, f"Target decoding branch {branch_name!r}"
            )
        return branches


DecoderComponentConfig = Annotated[
    Union[
        LinearDecoderComponentConfig,
        MLPDecoderComponentConfig,
        CompositeDecoderComponentConfig,
    ],
    Field(discriminator="type"),
]


class BackboneAttentionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["mha", "mqa", "gqa"] = "mha"
    n_heads: int = Field(..., gt=0)
    n_kv_heads: Optional[int] = Field(default=None, gt=0)
    output_projection: bool = True

    @model_validator(mode="after")
    def validate_heads(self):
        if self.n_kv_heads is None:
            if self.type == "gqa":
                raise ValueError("gqa requires n_kv_heads")
            self.n_kv_heads = 1 if self.type == "mqa" else self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads {self.n_heads} must be divisible by n_kv_heads "
                f"{self.n_kv_heads}"
            )
        if self.type == "mha" and self.n_kv_heads != self.n_heads:
            raise ValueError("mha requires n_kv_heads to equal n_heads")
        if self.type == "mqa" and self.n_kv_heads != 1:
            raise ValueError("mqa requires n_kv_heads to equal 1")
        return self


class BackboneFeedForwardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dim: int = Field(..., gt=0)
    activation: Literal["relu", "gelu", "swiglu"] = "swiglu"


class BackboneNormalizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["layer_norm", "rmsnorm"] = "rmsnorm"
    norm_first: bool = True


class BackbonePositionEncodingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["learned", "rope", "range", "sinusoidal"] = "learned"
    theta: float = Field(10000.0, gt=0.0)


class BackboneArchitectureConfig(BaseModel):
    """All and only fields that determine shared-backbone compatibility."""

    model_config = ConfigDict(extra="forbid")

    dim_model: int = Field(..., gt=0)
    max_context_length: int = Field(..., gt=0)
    num_layers: int = Field(..., gt=0)
    attention: BackboneAttentionConfig
    feed_forward: BackboneFeedForwardConfig
    normalization: BackboneNormalizationConfig = Field(
        default_factory=BackboneNormalizationConfig
    )
    position_encoding: BackbonePositionEncodingConfig = Field(
        default_factory=BackbonePositionEncodingConfig
    )
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    shared_layer_groups: list[list[int]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_architecture(self):
        n_heads = self.attention.n_heads
        if self.dim_model % n_heads != 0:
            raise ValueError(
                f"dim_model {self.dim_model} must be divisible by n_heads {n_heads}"
            )
        if self.position_encoding.type == "rope":
            head_dim = self.dim_model // n_heads
            if head_dim % 2 != 0:
                raise ValueError(
                    f"RoPE requires an even head dimension, got {head_dim}"
                )

        seen_layers: set[int] = set()
        for group in self.shared_layer_groups:
            if len(group) < 2 or len(group) != len(set(group)):
                raise ValueError(
                    "shared_layer_groups entries must contain at least two unique "
                    "layer indices"
                )
            invalid = [i for i in group if i < 0 or i >= self.num_layers]
            if invalid:
                raise ValueError(
                    "shared_layer_groups references indices outside the backbone: "
                    f"{invalid}"
                )
            overlap = seen_layers & set(group)
            if overlap:
                raise ValueError(
                    "shared_layer_groups cannot overlap: " f"{sorted(overlap)}"
                )
            seen_layers.update(group)
        return self


class BackboneRepositoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    load_policy: Literal["if_exists", "required"] = "if_exists"
    publish: bool = True
    conflict_policy: Literal["compare_and_swap"] = "compare_and_swap"


class BackboneComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: StrictStr = Field(..., min_length=1)
    architecture: BackboneArchitectureConfig
    repository: BackboneRepositoryConfig
    initialization: ModelInitializationConfig = Field(
        default_factory=ModelInitializationConfig
    )


def _validate_class_share_log_columns(config_values: dict[str, Any]) -> None:
    training_spec = config_values.get("training_spec", {})

    for col in training_spec.get("class_share_log_columns", []):
        if col not in config_values["target_columns"]:
            raise ValueError(f"Class-share column {col!r} must be a target column.")
        if config_values["target_column_types"].get(col) != "categorical":
            raise ValueError(
                f"Class-share column {col!r} must be a categorical target column."
            )
        if col not in config_values["n_classes"]:
            raise ValueError(
                f"Class-share column {col!r} has no configured class count."
            )
        if col not in config_values["id_maps"]:
            raise ValueError(
                f"Class-share column {col!r} has no index map for logging."
            )


@dataclass(frozen=True)
class LoadedTrainConfig:
    """A validated authored config together with its resolved runtime config."""

    config: "SequifierConfig"
    resolved: "ResolvedSequifierConfig"
    metadata: DatasetMetadata

    @property
    def source_values(self) -> dict[str, Any]:
        """Compatibility view used by the legacy partial-search compiler."""

        return self.config.model_dump(mode="python", exclude_unset=True)

    @property
    def model(self) -> "ResolvedSequifierConfig":
        """Compatibility alias for callers that previously consumed ``model``."""

        return self.resolved

    @property
    def metadata_values(self) -> dict[str, Any]:
        """Compatibility view of validated preprocessing metadata."""

        return self.metadata.model_dump(mode="python")


@beartype
def load_train_config_with_source(
    config_path: str, args_config: dict[str, Any], skip_metadata: bool
) -> LoadedTrainConfig:
    """Compose and validate authored YAML, then resolve preprocessing metadata."""
    config_values = load_composed_yaml_config(config_path)

    cli_values = {
        key: value for key, value in args_config.items() if key != "skip_metadata"
    }
    authored_values = merge_config_fragments((config_values, cli_values))
    authored_values, extracted_metadata_values = extract_inline_metadata(
        authored_values
    )
    inline_metadata_values = extracted_metadata_values if skip_metadata else None

    config = try_catch_excess_keys(config_path, SequifierConfig, authored_values)
    metadata_path = _effective_metadata_config_path(config)
    if skip_metadata:
        if inline_metadata_values is None:
            raise ValueError(
                "skip_metadata requires inline storage_layout, column data, class, "
                "and ID-map values so the config can still be resolved."
            )
        metadata = DatasetMetadata.model_validate(inline_metadata_values)
    else:
        if metadata_path is None:
            raise ValueError(
                f"Training config '{config_path}' must define metadata_config_path "
                "or preprocessing_data_path when metadata loading is enabled."
            )
        metadata = load_dataset_metadata(
            normalize_path(metadata_path, config.project_root)
        )

    resolved = resolve_sequifier_config(config, metadata)
    return LoadedTrainConfig(
        config=config,
        resolved=resolved,
        metadata=metadata,
    )


@beartype
def load_train_config(
    config_path: str, args_config: dict[str, Any], skip_metadata: bool
) -> "ResolvedSequifierConfig":
    """Load train YAML plus CLI overrides and optional metadata-derived fields."""
    return load_train_config_with_source(
        config_path,
        args_config,
        skip_metadata,
    ).resolved


def _effective_metadata_config_path(config: "SequifierConfig") -> Optional[str]:
    if config.metadata_config_path:
        return config.metadata_config_path
    if config.preprocessing_data_path:
        return metadata_config_path_from_preprocessing_data_path(
            config.preprocessing_data_path
        )
    return None


def resolve_sequifier_config(
    config: "SequifierConfig", metadata: DatasetMetadata
) -> "ResolvedSequifierConfig":
    """Resolve dataset metadata without mutating the validated authored config."""

    storage_layout = metadata.storage_layout
    if storage_layout.version != 2:
        raise ValueError(
            "Training requires metadata stored_window_layout_version=2, "
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
        raise ValueError("No columns found in resolved training config")

    target_column_types = config.target_column_types or derive_target_column_types(
        config.target_columns,
        column_data_types,
    )
    target_offset = target_offset_for_objective(
        config.training_objective,
        config.target_offset,
    )
    window_view = ModelWindowView(
        context_length=config.context_length,
        objective=config.training_objective,
        target_offset=target_offset,
    )
    resolve_window_view(storage_layout, window_view)

    if not metadata.split_paths and (
        config.data_path is None or config.validation_data_path is None
    ):
        raise ValueError(
            "Resolved training config needs data_path and validation_data_path "
            "when metadata does not provide split_paths."
        )
    data_path = config.data_path or metadata.split_paths[0]
    validation_data_path = (
        config.validation_data_path
        or metadata.split_paths[min(1, len(metadata.split_paths) - 1)]
    )

    resolved_values = config.model_dump(mode="python")
    resolved_values.update(
        {
            "metadata_config_path": _effective_metadata_config_path(config),
            "data_path": normalize_path(data_path, config.project_root),
            "validation_data_path": normalize_path(
                validation_data_path, config.project_root
            ),
            "input_columns": input_columns,
            "column_data_types": column_data_types,
            "categorical_columns": categorical_columns,
            "real_columns": real_columns,
            "target_column_types": target_column_types,
            "id_maps": metadata.id_maps,
            "special_token_ids": metadata.special_token_ids,
            "storage_layout": storage_layout,
            "window_view": window_view,
            "n_classes": metadata.n_classes,
        }
    )
    _validate_class_share_log_columns(resolved_values)
    return ResolvedSequifierConfig.model_validate(resolved_values)


class DotDict(dict):
    """Dot notation access to dictionary attributes."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore

    def __deepcopy__(self, memo=None):
        return DotDict(copy.deepcopy(dict(self), memo=memo))

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.update(state)


class ReplacementDistribution(BaseModel):
    masked: float = Field(..., ge=0.0, le=1.0)
    random: float = Field(..., ge=0.0, le=1.0)
    identical: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_sum(self):
        total = self.masked + self.random + self.identical
        if not math.isclose(total, 1.0, abs_tol=1e-5):
            raise ValueError(
                f"Replacement distribution probabilities must sum to 1.0, got {total}"
            )
        return self


class BERTSpecModel(BaseModel):
    masking_probability: float = Field(..., gt=0.0, le=1.0)
    replacement_distribution: ReplacementDistribution
    span_masking: ProbabilityDistribution


class NextOccurrenceConfigModel(BaseModel):
    column_name: str
    target_values: list[NextOccurrenceTargetValue] = Field(..., min_length=1)


class ResumeConfig(BaseModel):
    """Policy for resuming the same run from its complete checkpoint."""

    model_config = ConfigDict(extra="forbid")

    policy: Literal["never", "if_exists", "required"] = "never"
    checkpoint_path: Optional[str] = None

    @model_validator(mode="after")
    def validate_checkpoint_path(self):
        if self.policy == "required" and not self.checkpoint_path:
            raise ValueError(
                "training_spec.resume.checkpoint_path is required when policy is "
                "'required'"
            )
        return self


class TrainingSpecModel(BaseModel):
    """Training loop, optimization, precision, and distribution settings."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    device_max_concat_length: int = 12
    epochs: int
    log_interval: int = 10
    class_share_log_columns: list[str] = Field(default_factory=list)
    early_stopping_epochs: Optional[int] = None
    save_interval_epochs: int
    save_latest_interval_minutes: Optional[float] = None
    save_interval_minutes: Optional[float] = None
    save_interval_batches: Optional[int] = None
    save_interval_val_loss: bool = True
    calculate_validation_loss_on_initialization: bool = True
    batch_size: int
    learning_rate: float
    criterion: dict[str, str]
    class_weights: Optional[dict[str, list[float]]] = None
    accumulation_steps: Optional[int] = None
    gradient_clip: Optional[float] = None
    loss_weights: Optional[dict[str, float]] = None
    optimizer: DotDict = Field(default_factory=lambda: DotDict({"name": "Adam"}))
    scheduler: DotDict = Field(
        default_factory=lambda: DotDict(
            {"name": "StepLR", "step_size": 1, "gamma": 0.99}
        )
    )
    scheduler_step_on: str = "epoch"
    bert_spec: Optional[BERTSpecModel] = None
    next_occurrence_config: Optional[NextOccurrenceConfigModel] = None

    resume: ResumeConfig = Field(default_factory=ResumeConfig)
    enforce_determinism: bool = False
    distributed: bool = False
    load_full_data_to_ram: bool = True
    max_ram_gb: Union[int, float] = 16
    world_size: int = 1
    num_workers: int = 0
    backend: str = "nccl"
    layer_type_dtypes: Optional[dict[str, str]] = None
    layer_autocast: Optional[bool] = False
    data_parallelism: Optional[str] = None
    fsdp_cpu_offload: Optional[bool] = None
    torch_compile: str = "outer"
    float32_matmul_precision: str = "highest"

    def __init__(self, **kwargs):
        values = dict(kwargs)
        optimizer = values.pop("optimizer", {"name": "Adam"})
        scheduler = values.pop(
            "scheduler",
            {"name": "StepLR", "step_size": 1, "gamma": 0.99},
        )
        super().__init__(**values)

        self.optimizer = DotDict(self.validate_optimizer_config(optimizer))
        scheduler_context = {
            "epochs": self.epochs,
            "scheduler_step_on": self.scheduler_step_on,
        }
        self.scheduler = DotDict(
            self.validate_scheduler_config(scheduler, scheduler_context)
        )

    @field_serializer("optimizer", "scheduler")
    def serialize_dotdict(self, value: DotDict) -> dict[str, Any]:
        return dict(value)

    @field_validator("layer_type_dtypes")
    @classmethod
    def validate_layer_type_dtypes(cls, v):
        expected_keys = ["embedding", "linear", "conv", "norm", "decoder"]
        allowed_types = [
            "float32",
            "float16",
            "bfloat16",
            "float64",
            "float8_e4m3fn",
            "float8_e5m2",
        ]
        bad_keys, bad_types = [], []
        if v:
            for k, vv in v.items():
                if k not in expected_keys:
                    bad_keys.append(k)
                if vv not in allowed_types:
                    bad_types.append(vv)

            if len(bad_keys) > 0:
                raise ValueError(
                    f"The following keys are invalid: {bad_keys}. Allowed keys are: {expected_keys}"
                )

            if len(bad_types) > 0:
                raise ValueError(
                    f"The following layer types are invalid: {bad_types}. Allowed types are: {allowed_types}"
                )

        return v

    @field_validator("float32_matmul_precision")
    @classmethod
    def validate_float32_matmul_precision(cls, v):
        allowed_precisions = ["highest", "high", "medium"]
        if v not in allowed_precisions:
            raise ValueError(
                f"float32_matmul_precision must be one of {allowed_precisions}, got '{v}'"
            )
        return v

    @field_validator("criterion")
    @classmethod
    def validate_criterion(cls, v):
        for vv in v.values():
            if not hasattr(torch.nn, vv):
                raise ValueError(f"{vv} not in torch.nn")
        return v

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer_config(cls, v):
        if "name" not in v:
            raise ValueError("optimizer dict must specify 'name' field")
        if (
            not hasattr(torch.optim, v["name"])
            and not hasattr(torch_optimizer, v["name"])
            and not hasattr(sequifier.optimizers, v["name"])  # type: ignore
        ):
            raise ValueError(f"{v['name']} not in torch.optim or in torch_optimizer")
        return v

    @field_validator("scheduler")
    @classmethod
    def validate_scheduler_config(cls, v, info_dict):
        if "name" not in v:
            raise ValueError("scheduler dict must specify 'name' field")
        if not hasattr(torch.optim.lr_scheduler, v["name"]):
            raise ValueError(f"{v} not in torch.optim.lr_scheduler")
        if "total_steps" in v:
            if info_dict.get("scheduler_step_on") == "epoch":
                if not v["total_steps"] == info_dict.get("epochs"):
                    raise ValueError(
                        f"scheduler total steps: {v['total_steps']} != {info_dict.get('epochs')}: total epochs"
                    )
            else:
                logger.info(
                    f"[WARNING] {v['total_steps']} scheduler steps at {info_dict.get('epochs')} epochs implies {v['total_steps']/info_dict.get('epochs'):.2f} batches. Does this seem correct?"
                )
        return v

    @field_validator("scheduler_step_on")
    @classmethod
    def validate_scheduler_step_on(cls, v):
        if v not in ["epoch", "batch"]:
            raise ValueError(
                f"scheduler_step_on must be in ['epoch', 'batch'], {v} isn't"
            )
        return v

    @field_validator("data_parallelism")
    @classmethod
    def validate_data_parallelism(cls, v):
        if v is not None and v not in ["DDP", "FSDP"]:
            raise ValueError(
                f"data_parallelism must be None, or 'DDP' or 'FSDP', got '{v}'"
            )
        return v


class ModelSpecModel(BaseModel):
    """The three independently configured runtime model components."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    ingestion: IngestionComponentConfig
    backbone: BackboneComponentConfig
    decoder: DecoderComponentConfig

    @model_validator(mode="after")
    def validate_component_contracts(self):
        dim_model = self.backbone.architecture.dim_model
        if (
            self.ingestion.type != "composite"
            and self.ingestion.output_dim != dim_model
        ):
            raise ValueError(
                "model_spec.ingestion.output_dim must equal "
                "model_spec.backbone.architecture.dim_model: "
                f"{self.ingestion.output_dim} != {dim_model}"
            )
        return self


_PathT = TypeVar("_PathT")
_InputColumnsT = TypeVar("_InputColumnsT")
_ColumnTypesT = TypeVar("_ColumnTypesT")


class _SequifierConfigBase(BaseModel, Generic[_PathT, _InputColumnsT, _ColumnTypesT]):
    """Shared fields and validation for authored and resolved training config."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    project_root: str
    preprocessing_data_path: Optional[str] = None
    metadata_config_path: _PathT = Field(default=None)
    model_name: str
    training_objective: str
    device: str
    data_path: _PathT = Field(default=None)
    validation_data_path: _PathT = Field(default=None)
    read_format: str = "parquet"

    input_columns: _InputColumnsT
    column_data_types: _ColumnTypesT = Field(default=None)
    target_columns: list[str]
    target_column_types: _ColumnTypesT = Field(default=None)
    categorical_decoder_special_tokens: dict[
        str, list[Literal["unknown", "other", "mask"]]
    ] = Field(default_factory=dict)

    context_length: int = Field(gt=0)
    target_offset: int = Field(default=1, ge=0)
    model_window_stride: Optional[int] = Field(default=None, gt=0)
    inference_batch_size: int
    seed: int = 1010

    export_generative_model: bool
    export_embedding_model: bool
    export_onnx: bool = True
    export_pt: bool = False
    export_with_dropout: bool = False

    feature_layout: Optional[FeatureLayoutRegistryModel] = None
    model_spec: ModelSpecModel
    training_spec: TrainingSpecModel

    @field_validator("training_objective")
    @classmethod
    def validate_authored_training_objective(cls, value: str) -> str:
        if value not in ALLOWED_OBJECTIVE_NAMES:
            raise ValueError(
                f"Only {OBJECTIVE_NAME_MESSAGE} are allowed, found {value}"
            )
        return value

    @field_validator("model_name")
    @classmethod
    def validate_authored_model_name(cls, value: str) -> str:
        if "embedding" in value:
            raise ValueError("model_name cannot contain 'embedding'")
        return value

    @field_validator("read_format")
    @classmethod
    def validate_authored_read_format(cls, value: str) -> str:
        if value not in {"csv", "parquet", "pt"}:
            raise ValueError("Currently only 'csv', 'parquet' and 'pt' are supported")
        return value

    @field_validator("target_column_types")
    @classmethod
    def validate_authored_target_column_types(cls, value, info):
        if value is None:
            return value
        if any(
            type_name not in {"categorical", "real"} for type_name in value.values()
        ):
            raise ValueError("Target column types must be either categorical or real")
        if list(value) != info.data.get("target_columns", []):
            raise ValueError(
                "target_columns and target_column_types must contain the same "
                "values/keys in the same order"
            )
        return value

    @model_validator(mode="after")
    def validate_authored_relationships(self):
        if self.metadata_config_path is None and self.preprocessing_data_path is None:
            raise ValueError(
                "metadata_config_path is required when preprocessing_data_path "
                "is not provided"
            )

        objective_class = get_objective_class(self.training_objective)
        is_bert = issubclass(objective_class, BERTObjective)
        is_next_occurrence = issubclass(objective_class, NextOccurrenceObjective)
        if self.training_spec.bert_spec is not None and not is_bert:
            raise ValueError(
                "The BERT hyperparameters should only be configured if the "
                "training objective is 'bert'"
            )
        if self.training_spec.bert_spec is None and is_bert:
            raise ValueError(
                "If the training_objective is 'bert', the BERT hyperparameters "
                "must be set"
            )
        if (
            self.training_spec.next_occurrence_config is not None
            and not is_next_occurrence
        ):
            raise ValueError(
                "next_occurrence_config should only be configured if the "
                "training objective is 'next_occurrence'"
            )
        if self.training_spec.next_occurrence_config is None and is_next_occurrence:
            raise ValueError(
                "If the training_objective is 'next_occurrence', "
                "next_occurrence_config must be set"
            )

        effective_target_offset = target_offset_for_objective(
            self.training_objective, self.target_offset
        )
        ModelWindowView(
            context_length=self.context_length,
            objective=self.training_objective,
            target_offset=effective_target_offset,
        )
        objective_class.validate_prediction_length(
            self.model_spec.decoder.prediction_length,
            self.context_length,
            usage="training",
        )
        max_context_length = self.model_spec.backbone.architecture.max_context_length
        if self.context_length > max_context_length:
            raise ValueError(
                f"context_length {self.context_length} exceeds backbone "
                f"max_context_length {max_context_length}"
            )
        decoded_context_length = (
            self.context_length - self.model_spec.decoder.support + 1
        )
        if self.model_spec.decoder.support > self.context_length:
            raise ValueError("model_spec.decoder.support cannot exceed context_length")
        if self.model_spec.decoder.prediction_length > decoded_context_length:
            raise ValueError(
                "model_spec.prediction_length cannot exceed the number of "
                "decoded positions produced by decoding_support"
            )
        if set(self.training_spec.criterion) != set(self.target_columns):
            raise ValueError(
                "target_columns and criterion must contain the same values/keys"
            )
        if (
            not self.export_generative_model
            and not self.export_embedding_model
            and os.getenv("SEQUIFIER_PREVENT_EXPORT") is None
        ):
            raise ValueError(
                "At least one of export_generative_model and "
                "export_embedding_model must be true"
            )
        return self


class SequifierConfig(
    _SequifierConfigBase[Optional[str], Optional[list[str]], Optional[dict[str, str]]]
):
    """User-authored configuration for one concrete training run."""


class ResolvedSequifierConfig(_SequifierConfigBase[str, list[str], dict[str, str]]):
    """Internal training config after preprocessing metadata has been resolved."""

    metadata_config_path: str
    data_path: str
    validation_data_path: str
    input_columns: list[str]
    column_data_types: dict[str, str]
    target_column_types: dict[str, str]
    categorical_columns: list[str]
    real_columns: list[str]
    id_maps: dict[str, dict[str | int, int]]
    special_token_ids: dict[str, int] = Field(
        default_factory=lambda: SPECIAL_TOKEN_IDS.ids_by_label
    )
    storage_layout: StoredWindowLayout
    window_view: ModelWindowView
    n_classes: dict[str, int]

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

    @field_validator("training_objective")
    @classmethod
    def validate_training_objective(cls, v):
        if v not in ALLOWED_OBJECTIVE_NAMES:
            raise ValueError(f"Only {OBJECTIVE_NAME_MESSAGE} are allowed, found {v}")
        return v

    @model_validator(mode="after")
    def validate_objective_specific_config(self):
        objective_class = get_objective_class(self.training_objective)
        is_bert_objective = issubclass(objective_class, BERTObjective)
        is_next_occurrence_objective = issubclass(
            objective_class,
            NextOccurrenceObjective,
        )
        bert_spec = self.training_spec.bert_spec
        next_occurrence_config = self.training_spec.next_occurrence_config
        if bert_spec is not None and not is_bert_objective:
            raise ValueError(
                "The BERT hyperparameters should only be configured if the "
                "training objective is 'bert'"
            )
        if bert_spec is None and is_bert_objective:
            raise ValueError(
                "If the training_objective is 'bert', the BERT hyperparameters "
                "must be set"
            )
        if next_occurrence_config is not None and not is_next_occurrence_objective:
            raise ValueError(
                "next_occurrence_config should only be configured if the "
                "training objective is 'next_occurrence'"
            )
        if next_occurrence_config is None and is_next_occurrence_objective:
            raise ValueError(
                "If the training_objective is 'next_occurrence', "
                "next_occurrence_config must be set"
            )
        return self

    @field_validator("special_token_ids")
    @classmethod
    def validate_special_token_ids_match_runtime(cls, v):
        return validate_special_token_ids(v, source="TrainModel")

    @field_validator("categorical_decoder_special_tokens")
    @classmethod
    def validate_decoder_special_token_lists(cls, v):
        if any(len(tokens) != len(set(tokens)) for tokens in v.values()):
            raise ValueError(
                "categorical_decoder_special_tokens cannot contain duplicate tokens."
            )
        return {
            column: [name for name in SPECIAL_TOKEN_NAMES if name in tokens]
            for column, tokens in v.items()
        }

    @model_validator(mode="after")
    def validate_decoder_special_token_columns(self):
        target_column_types = self.target_column_types
        if target_column_types is None:
            raise ValueError("target_column_types must be provided or derived")
        categorical_targets = {
            col
            for col in self.target_columns
            if target_column_types[col] == "categorical"
        }
        invalid = set(self.categorical_decoder_special_tokens) - categorical_targets
        if invalid:
            raise ValueError(
                "categorical_decoder_special_tokens may only reference categorical "
                f"target columns, found {sorted(invalid)}."
            )
        resolve_categorical_decoder_ids(
            self.target_columns,
            target_column_types,
            self.n_classes,
            self.categorical_decoder_special_tokens,
        )
        return self

    @model_validator(mode="after")
    def validate_bert_prediction_length_matches_context_length(self):
        if self.window_view.objective != self.training_objective:
            raise ValueError(
                "window_view objective must match training_objective "
                f"({self.window_view.objective} != {self.training_objective})."
            )
        resolve_window_view(self.storage_layout, self.window_view)
        get_objective_class(self.training_objective).validate_prediction_length(
            self.model_spec.decoder.prediction_length,
            self.window_view.context_length,
            usage="training",
        )
        return self

    @model_validator(mode="after")
    def validate_next_occurrence_config_matches_targets(self):
        target_column_types = self.target_column_types
        if target_column_types is None:
            raise ValueError("target_column_types must be provided or derived")
        objective_class = get_objective_class(self.training_objective)
        if issubclass(objective_class, NextOccurrenceObjective):
            next_occurrence_config = self.training_spec.next_occurrence_config
            if next_occurrence_config is None:
                raise ValueError(
                    "next_occurrence_config must be set for next_occurrence training."
                )

            column_name = next_occurrence_config.column_name
            if column_name not in self.target_columns:
                raise ValueError(
                    "next_occurrence_config.column_name must be one of target_columns, "
                    f"got {column_name!r}."
                )
            if target_column_types.get(column_name) != "categorical":
                raise ValueError(
                    "next_occurrence_config.column_name must refer to a categorical target column."
                )
            if column_name not in self.id_maps:
                raise ValueError(
                    "next_occurrence_config.column_name must have a preprocessing id_map, "
                    f"got {column_name!r}."
                )

            id_map = self.id_maps[column_name]
            missing_values = [
                value
                for value in next_occurrence_config.target_values
                if value not in id_map
            ]
            if missing_values:
                raise ValueError(
                    "next_occurrence_config.target_values must match keys in "
                    f"id_maps[{column_name!r}] exactly, missing {missing_values!r}."
                )
        return self

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v):
        if not "embedding" not in v:
            raise ValueError("model_name cannot contain 'embedding'")
        return v

    @field_validator("target_column_types")
    @classmethod
    def validate_target_column_types(cls, v, info):
        if not all(vv in ["categorical", "real"] for vv in v.values()):
            raise ValueError(
                f"Invalid target_column_types found: {[vv not in ['categorical', 'real'] for vv in v.values()]}. Only 'categorical' and 'real' are allowed."
            )
        if not (list(v.keys()) == info.data.get("target_columns")):
            raise ValueError(
                "target_columns and target_column_types must contain the same values/keys in the same order"
            )
        return v

    @field_validator("read_format")
    @classmethod
    def validate_read_format(cls, v):
        if v not in [
            "csv",
            "parquet",
            "pt",
        ]:
            raise ValueError("Currently only 'csv', 'parquet' and 'pt' are supported")
        return v

    @field_validator("training_spec")
    @classmethod
    def validate_training_spec(cls, v, info):
        if not set(info.data.get("target_columns")) == set(v.criterion.keys()):
            raise ValueError(
                "target_columns and criterion must contain the same values/keys"
            )

        if v.distributed:
            if info.data.get("read_format") not in ["pt", "parquet"]:
                raise ValueError(
                    "If distributed is set to 'true', the format must be 'pt' or 'parquet' representing a folder dataset."
                )
            if info.data.get("read_format") == "parquet":
                warnings.warn(
                    "Training on distributed data in parquet format takes significantly more CPU per GPU than with 'pt'."
                )

        if (
            v.save_latest_interval_minutes is not None
            and not os.getenv("SEQUIFIER_TESTING", "0") == "1"
            and v.save_latest_interval_minutes == 0
        ):
            raise ValueError("save_latest_interval_minutes must be larger than 0")

        if (
            v.save_interval_minutes is not None
            and not os.getenv("SEQUIFIER_TESTING", "0") == "1"
            and v.save_interval_minutes == 0
        ):
            raise ValueError("save_interval_minutes must be larger than 0")

        if (
            v.save_interval_batches is not None
            and not os.getenv("SEQUIFIER_TESTING", "0") == "1"
            and v.save_interval_batches == 0
        ):
            raise ValueError("save_interval_batches must be larger than 0")

        if v.torch_compile not in ["outer", "inner", "none"]:
            raise ValueError(
                f'torch_compile {v.torch_compile} invalid, must be one of ["outer", "inner", "none"]'
            )

        if v.data_parallelism == "FSDP":
            if v.layer_type_dtypes is not None:
                raise ValueError(
                    "FSDP does not support manual layer pre-casting. Please set "
                    "'layer_type_dtypes' to null when using FSDP, and rely on "
                    "'layer_autocast' (MixedPrecisionPolicy) instead."
                )
            if v.fsdp_cpu_offload is None:
                raise ValueError(
                    "If data_parallelism == 'FSDP', fsdp_cpu_offload cannot be None"
                )

        if v.data_parallelism == "FSDP" and v.torch_compile == "outer":
            raise ValueError(
                "If data_parallelism is set to 'FSDP' then torch_compile must be one of 'none' and 'inner'"
            )

        if v.data_parallelism == "DDP" and v.torch_compile == "inner":
            raise ValueError(
                "If data_parallelism is set to 'DDP' then torch_compile must be one of 'none' and 'outer'"
            )

        if v.data_parallelism is None or v.data_parallelism != "FSDP":
            if v.fsdp_cpu_offload is not None:
                raise ValueError(
                    "If data_parallelism != 'FSDP', fsdp_cpu_offload must be None"
                )
        if v.data_parallelism == "FSDP":
            if v.fsdp_cpu_offload is None:
                raise ValueError(
                    "If data_parallelism == 'FSDP', fsdp_cpu_offload cannot be None"
                )

        if v.distributed and v.data_parallelism is None:
            raise ValueError(
                "If 'distributed' is True, data_parallelism cannot be 'None'"
            )

        export_generative_model = info.data.get("export_generative_model")
        export_embedding_model = info.data.get("export_embedding_model")
        if (
            not export_generative_model
            and not export_embedding_model
            and os.getenv("SEQUIFIER_PREVENT_EXPORT") is None
        ):
            raise ValueError(
                "At least one of 'export_generative_model' and 'export_embedding_model' must be true. If you want to override this, set the env variable 'SEQUIFIER_PREVENT_EXPORT' to any value"
            )

        return v

    @field_validator("column_data_types")
    @classmethod
    def validate_column_types(cls, v, info):
        target_columns = info.data.get("target_columns", [])
        column_ordered = list(v.keys())
        columns_ordered_filtered = [c for c in column_ordered if c in target_columns]
        if not (columns_ordered_filtered == target_columns):
            raise ValueError(f"{columns_ordered_filtered = } != {target_columns = }")
        return v

    @model_validator(mode="after")
    def validate_feature_layout_columns(self):
        if self.feature_layout is None:
            return self

        allowed_columns = set(self.input_columns)
        for layout_name, layout in self.feature_layout.items():
            missing_columns = set(layout.columns) - allowed_columns
            if missing_columns:
                raise ValueError(
                    f"feature_layout {layout_name!r} references unknown columns: "
                    f"{sorted(missing_columns)}"
                )

        return self

    @model_validator(mode="after")
    def validate_auxiliary_input_columns(self):
        auxiliary_columns = set(self.model_spec.ingestion.auxiliary_input_columns)
        missing_columns = auxiliary_columns - set(self.input_columns)
        if missing_columns:
            raise ValueError(
                "model_spec.ingestion.auxiliary_input_columns references unknown input "
                f"columns: {sorted(missing_columns)}"
            )

        return self

    @model_validator(mode="after")
    def validate_decoder(self):
        decoder_support = self.model_spec.decoder.support
        context_length = self.window_view.context_length
        if decoder_support > context_length:
            raise ValueError(
                "model_spec.decoder.support must be in the range "
                f"[1, context_length], got support={decoder_support} "
                f"and context_length={context_length}."
            )

        decoded_context_length = context_length - decoder_support + 1
        if self.model_spec.decoder.prediction_length > decoded_context_length:
            raise ValueError(
                "model_spec.decoder.prediction_length cannot exceed the number of "
                "decoded positions produced by decoder support. Got "
                f"prediction_length={self.model_spec.decoder.prediction_length}, "
                f"decoded_context_length={decoded_context_length}, "
                f"support={decoder_support}."
            )

        from sequifier.model.decoders import resolve_decoding_plan

        resolve_decoding_plan(self)
        return self

    @model_validator(mode="after")
    def validate_ingestion(self):
        # Keep public configuration models independent from runtime modules while
        # using the same resolver for cross-field validation and construction.
        # The local import avoids a config/model import cycle at module load time.
        from sequifier.model.ingestion_compiler import resolve_ingestion_plan

        resolve_ingestion_plan(self)
        return self


# Compatibility name for integrations that construct or annotate the historical
# resolved training model directly.  New authored config code should use
# ``SequifierConfig`` and resolve it explicitly.
TrainModel = ResolvedSequifierConfig
