"""Typed configuration for model-parameter initialization overrides."""

from typing import Annotated, Literal, TypeAlias, Union

from pydantic import Field, RootModel, field_serializer, model_validator

from sequifier.config.layer_groups import LayerGroup
from sequifier.config.naming import ConfigModel
from sequifier.typechecking import beartype

ParameterKind: TypeAlias = Literal["weight", "bias"]
InitializationTarget: TypeAlias = tuple[LayerGroup, ParameterKind]


class _InitializationMethod(ConfigModel):
    """Base class for strictly validated initialization methods."""


class PreserveInitialization(_InitializationMethod):
    method: Literal["preserve"]


class NormalInitialization(_InitializationMethod):
    method: Literal["normal"]
    mean: float = 0.0
    std: float = Field(..., ge=0.0)


class UniformInitialization(_InitializationMethod):
    method: Literal["uniform"]
    low: float
    high: float

    @model_validator(mode="after")
    @beartype
    def validate_bounds(self):
        if self.low > self.high:
            raise ValueError(
                f"uniform low must be <= high, got {self.low} > {self.high}"
            )
        return self


class XavierUniformInitialization(_InitializationMethod):
    method: Literal["xavier_uniform"]
    gain: float = Field(1.0, ge=0.0)
    fan_mode: Literal["per_tensor", "joint"] = "per_tensor"

    @field_serializer("method")
    @beartype
    def serialize_method(self, method):
        return "xavier_uniform"


class XavierNormalInitialization(_InitializationMethod):
    method: Literal["xavier_normal"]
    gain: float = Field(1.0, ge=0.0)
    fan_mode: Literal["per_tensor", "joint"] = "per_tensor"

    @field_serializer("method")
    @beartype
    def serialize_method(self, method):
        return "xavier_normal"


class _KaimingInitialization(_InitializationMethod):
    a: float = 0.0
    mode: Literal["fan_in", "fan_out"] = "fan_in"
    nonlinearity: Literal[
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "sigmoid",
        "tanh",
        "relu",
        "leaky_relu",
        "selu",
    ] = "leaky_relu"


class KaimingUniformInitialization(_KaimingInitialization):
    method: Literal["kaiming_uniform"]


class KaimingNormalInitialization(_KaimingInitialization):
    method: Literal["kaiming_normal"]


class ConstantInitialization(_InitializationMethod):
    method: Literal["constant"]
    value: float


class ZerosInitialization(_InitializationMethod):
    method: Literal["zeros"]


class OnesInitialization(_InitializationMethod):
    method: Literal["ones"]


class IdentityPlusNormalInitialization(_InitializationMethod):
    """Identity map in all but the final input column, which is normal."""

    method: Literal["identity_plus_normal"]
    mean: float = 0.0
    std: float = Field(0.02, ge=0.0)


InitializationMethodConfig: TypeAlias = Annotated[
    Union[
        PreserveInitialization,
        NormalInitialization,
        UniformInitialization,
        XavierUniformInitialization,
        XavierNormalInitialization,
        KaimingUniformInitialization,
        KaimingNormalInitialization,
        ConstantInitialization,
        ZerosInitialization,
        OnesInitialization,
        IdentityPlusNormalInitialization,
    ],
    Field(discriminator="method"),
]


class LayerGroupInitialization(ConfigModel):
    """Optional weight and bias overrides for one semantic layer group."""

    weight: InitializationMethodConfig | None = None
    bias: InitializationMethodConfig | None = None

    def parameter_kinds(self) -> tuple[ParameterKind, ...]:
        kinds: list[ParameterKind] = []
        if self.weight is not None:
            kinds.append("weight")
        if self.bias is not None:
            kinds.append("bias")
        return tuple(kinds)

    @model_validator(mode="after")
    @beartype
    def validate_not_empty(self):
        if self.weight is None and self.bias is None:
            raise ValueError("layer initialization must configure weight or bias")
        return self


class ModelInitializationConfig(RootModel[dict[LayerGroup, LayerGroupInitialization]]):
    """Direct mapping from semantic layer groups to initialization overrides."""

    root: dict[LayerGroup, LayerGroupInitialization] = Field(default_factory=dict)

    @beartype
    def override_for(self, group: LayerGroup) -> LayerGroupInitialization | None:
        return self.root.get(group)

    @beartype
    def configured_targets(self) -> set[InitializationTarget]:
        return {
            (group, parameter_kind)
            for group, initialization in self.root.items()
            for parameter_kind in initialization.parameter_kinds()
        }
