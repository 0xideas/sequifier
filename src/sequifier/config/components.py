"""Reusable configuration components shared by training and artifacts."""

import math
from collections.abc import Mapping
from itertools import product
from typing import Annotated, Any, Literal, Optional, TypeAlias, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StrictInt,
    StrictStr,
    field_validator,
    model_serializer,
    model_validator,
)

from sequifier.config.initialization_config import ModelInitializationConfig
from sequifier.config.probabilities import ProbabilityDistribution
from sequifier.typechecking import beartype

AnyType = str | int | float
NextOccurrenceTargetValue: TypeAlias = StrictInt | StrictStr


class CartesianLayoutModel(BaseModel):
    """Reusable coordinate annotation for flat feature columns."""

    model_config = ConfigDict(extra="forbid")

    axes: dict[str, list[AnyType]] = Field(..., min_length=1)
    columns: dict[str, dict[str, AnyType]] = Field(..., min_length=1)

    @model_validator(mode="after")
    @beartype
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

    @beartype
    def items(self):
        return self.root.items()

    @beartype
    def __contains__(self, key: str) -> bool:
        return key in self.root

    @beartype
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
    @beartype
    def validate_auxiliary_input_columns(cls, value):
        _validate_column_list_unique(value, "model.ingestion.auxiliary_input_columns")
        return value


class EmbeddingIngestionConfig(IngestionComponentBase):
    """Use the existing flat-column embedding path."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["embedding"] = "embedding"
    columns: Optional[list[str]] = Field(default=None, min_length=1)
    output_dim: int = Field(..., gt=0)
    feature_embedding_dims: Optional[dict[str, int]] = None

    @field_validator("columns")
    @classmethod
    @beartype
    def validate_columns(cls, v):
        if v is not None:
            _validate_column_list_unique(v, "embedding ingestion columns")
        return v

    @field_validator("feature_embedding_dims")
    @classmethod
    @beartype
    def validate_feature_embedding_dims(cls, v):
        _validate_feature_embedding_dims(v, "embedding feature_embedding_dims")
        return v


class PassthroughIngestionConfig(IngestionComponentBase):
    """Pass real-valued columns through without per-feature encoders."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["passthrough"]
    columns: Optional[list[str]] = Field(default=None, min_length=1)
    output_dim: int = Field(..., gt=0)

    @field_validator("columns")
    @classmethod
    @beartype
    def validate_columns(cls, v):
        if v is not None:
            _validate_column_list_unique(v, "passthrough ingestion columns")
        return v


class FeaturePoolIngestionConfig(IngestionComponentBase):
    """Encode columns as feature tokens before pooling to one time token."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["feature_pool"]
    columns: list[str] = Field(..., min_length=1)
    output_dim: int = Field(..., gt=0)

    @field_validator("columns")
    @classmethod
    @beartype
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
    @beartype
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
    @beartype
    def validate_columns(cls, v):
        _validate_column_list_unique(v, "siamese ingestion columns")
        return v


class TemporalConvIngestionConfig(IngestionComponentBase):
    """Encode columns, then apply Conv1D over the time axis."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["temporal_conv"]
    columns: list[str] = Field(..., min_length=1)
    output_dim: int = Field(..., gt=0)
    base_ingestion: Literal["embedding", "passthrough"] = "embedding"
    feature_embedding_dims: Optional[dict[str, int]] = None
    kernel_size: int = Field(3, gt=0)
    dilation: int | list[int] = 1
    num_layers: int = Field(1, gt=0)
    causal: bool = True
    activation: Literal["relu", "gelu", "silu"] = "gelu"
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    post_conv_norm: Literal["layer_norm", "rmsnorm", "none"] = "layer_norm"
    orientation: Literal["within_column", "within_item_position"] = (
        "within_item_position"
    )

    @model_validator(mode="before")
    @classmethod
    @beartype
    def default_num_layers_from_dilation_schedule(cls, values):
        if not isinstance(values, dict):
            return values
        dilation = values.get("dilation")
        if isinstance(dilation, list) and "num_layers" not in values:
            values = dict(values)
            values["num_layers"] = len(dilation)
        return values

    @model_validator(mode="after")
    @beartype
    def validate_temporal_conv(self):
        _validate_column_list_unique(self.columns, "temporal_conv ingestion columns")
        _validate_feature_embedding_dims(
            self.feature_embedding_dims, "temporal_conv feature_embedding_dims"
        )
        if self.base_ingestion == "passthrough" and self.feature_embedding_dims:
            raise ValueError(
                "temporal_conv feature_embedding_dims is only valid when "
                "base_ingestion is 'embedding'"
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
    @beartype
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
    @beartype
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
    @beartype
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
    n_heads: int = Field(1, gt=0)
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    unshared_axes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    @beartype
    def validate_axes_unique(self):
        _validate_axis_list_unique(self.axes, "axes")
        _validate_axis_list_unique(self.unshared_axes, "unshared_axes")
        if self.output_dim % self.n_heads != 0:
            raise ValueError("axis_attention output_dim must be divisible by n_heads")
        return self


class AxisPoolBlockModel(BaseModel):
    """Reduce configured cartesian axes without changing the channel dimension."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["axis_pool"]
    axes: list[str] = Field(..., min_length=1)
    mode: Literal["mean", "sum", "max"] = "mean"

    @model_validator(mode="after")
    @beartype
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


@beartype
def _validate_axis_list_unique(axes: list[str], field_name: str) -> None:
    if len(axes) != len(set(axes)):
        raise ValueError(f"{field_name} cannot contain duplicate axes")


@beartype
def _validate_column_list_unique(columns: list[str], field_name: str) -> None:
    if len(columns) != len(set(columns)):
        raise ValueError(f"{field_name} cannot contain duplicate columns")


@beartype
def _validate_module_dict_key(key: str, usage: str) -> None:
    if key == "":
        raise ValueError(f"{usage} cannot be empty")
    if "." in key:
        raise ValueError(f"{usage} cannot contain '.'")


@beartype
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
    @beartype
    def normalize_type(cls, v):
        if v is None:
            return "none"
        if isinstance(v, str):
            return v.lower()
        return v

    @model_validator(mode="after")
    @beartype
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
    @beartype
    def normalize_axis_embeddings(cls, v):
        if v is None:
            return {"type": "none", "axes": []}
        if isinstance(v, list):
            return {"type": "learned", "axes": v}
        return v


BranchIngestionConfig = Annotated[
    Union[
        EmbeddingIngestionConfig,
        PassthroughIngestionConfig,
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
    @beartype
    def validate_branch_names(cls, branches):
        for branch_name in branches:
            _validate_module_dict_key(
                branch_name, f"Composite ingestion branch {branch_name!r}"
            )
        return branches


IngestionComponentConfig = Annotated[
    Union[
        EmbeddingIngestionConfig,
        PassthroughIngestionConfig,
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
    @beartype
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
    activation: Literal["relu", "gelu", "silu"] = "relu"
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    hidden_weight_l2: float = Field(
        0.0,
        ge=0.0,
        allow_inf_nan=False,
    )

    @field_validator("target_columns")
    @classmethod
    @beartype
    def validate_target_columns(cls, v):
        if v is not None:
            _validate_column_list_unique(v, "mlp decoding target_columns")
        return v

    @field_validator("hidden_dims")
    @classmethod
    @beartype
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
    @beartype
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
    @beartype
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

    type: Literal["learned", "rope", "range", "range_concat", "sinusoidal"] = "learned"
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
    positional_encoding_scope: Literal["per_feature", "global"] = "per_feature"
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    shared_layer_groups: list[list[int]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    @beartype
    def default_global_position_scope(cls, values):
        if not isinstance(values, dict):
            return values
        position_encoding = values.get("position_encoding", {})
        position_type = (
            position_encoding.get("type", "learned")
            if isinstance(position_encoding, dict)
            else getattr(position_encoding, "type", "learned")
        )
        if (
            position_type in {"range", "range_concat", "sinusoidal"}
            and "positional_encoding_scope" not in values
        ):
            values = dict(values)
            values["positional_encoding_scope"] = "global"
        return values

    @model_validator(mode="after")
    @beartype
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
        if (
            self.position_encoding.type in {"range", "range_concat", "sinusoidal"}
            and self.positional_encoding_scope != "global"
        ):
            raise ValueError(
                f"position_encoding type {self.position_encoding.type!r} requires "
                "positional_encoding_scope 'global'"
            )
        if self.position_encoding.type == "range_concat" and self.dim_model < 2:
            raise ValueError("range_concat requires dim_model to be at least 2")

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

    backbone_id: StrictStr = Field(..., min_length=1)
    path: str
    load_policy: Literal["if_exists", "required"] = "if_exists"
    publish: bool = True
    conflict_policy: Literal["compare_and_swap"] = "compare_and_swap"


class BackboneComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture: BackboneArchitectureConfig
    repository: Optional[BackboneRepositoryConfig] = None
    initialization: ModelInitializationConfig = Field(
        default_factory=ModelInitializationConfig
    )


class ComponentSpec(BaseModel):
    """A named component and its constructor arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    @beartype
    def parse_flattened(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("component configuration must be a mapping")
        if "arguments" in value:
            return value
        return {
            "name": value.get("name"),
            "arguments": {key: item for key, item in value.items() if key != "name"},
        }

    @model_serializer
    @beartype
    def serialize_flattened(self) -> dict[str, Any]:
        return {"name": self.name, **self.arguments}


class ReplacementDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    masked: float = Field(..., ge=0.0, le=1.0)
    random: float = Field(..., ge=0.0, le=1.0)
    identical: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    @beartype
    def validate_sum(self):
        total = self.masked + self.random + self.identical
        if not math.isclose(total, 1.0, abs_tol=1e-5):
            raise ValueError(
                f"Replacement distribution probabilities must sum to 1.0, got {total}"
            )
        return self


class BERTSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    masking_probability: float = Field(..., gt=0.0, le=1.0)
    replacement_distribution: ReplacementDistribution
    span_masking: ProbabilityDistribution


class NextOccurrenceConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column_name: str
    target_values: list[NextOccurrenceTargetValue] = Field(..., min_length=1)


class ResumeConfig(BaseModel):
    """Policy for resuming the same run from its complete checkpoint."""

    model_config = ConfigDict(extra="forbid")

    policy: Literal["never", "if_exists", "required"] = "never"
    checkpoint_path: Optional[str] = None

    @model_validator(mode="after")
    @beartype
    def validate_checkpoint_path(self):
        if self.policy == "required" and not self.checkpoint_path:
            raise ValueError(
                "global_training.resume.checkpoint_path is required when "
                "policy is 'required'"
            )
        return self
