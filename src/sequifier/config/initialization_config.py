"""Typed configuration for model-parameter initialization overrides."""

import math
from typing import Annotated, Any, Literal, TypeAlias, Union

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    RootModel,
    field_serializer,
    model_validator,
)

from sequifier.config.layer_groups import LayerGroup
from sequifier.typechecking import beartype

ParameterKind: TypeAlias = Literal["weight", "bias"]
InitializationTarget: TypeAlias = tuple[LayerGroup, ParameterKind]


class _InitializationMethod(BaseModel):
    """Base class for strictly validated initialization methods."""

    model_config = ConfigDict(extra="forbid")


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
    method: Literal[
        "xavier",
        "glorot",
        "xavier_uniform",
        "glorot_uniform",
    ]
    gain: float = Field(1.0, ge=0.0)
    fan_mode: Literal["per_tensor", "joint"] = "per_tensor"

    @field_serializer("method")
    @beartype
    def serialize_method(self, method):
        return "xavier_uniform"


class XavierNormalInitialization(_InitializationMethod):
    method: Literal["xavier_normal", "glorot_normal"]
    gain: float = Field(1.0, ge=0.0)
    fan_mode: Literal["per_tensor", "joint"] = "per_tensor"

    @field_serializer("method")
    @beartype
    def serialize_method(self, method):
        return "xavier_normal"


class KaimingUniformInitialization(_InitializationMethod):
    method: Literal["kaiming_uniform"]
    a: float = 0.0
    mode: Literal["fan_in", "fan_out"] = "fan_in"
    nonlinearity: str = "leaky_relu"


class KaimingNormalInitialization(_InitializationMethod):
    method: Literal["kaiming_normal"]
    a: float = 0.0
    mode: Literal["fan_in", "fan_out"] = "fan_in"
    nonlinearity: str = "leaky_relu"


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


@beartype
def _expand_method_shorthand(value: Any) -> Any:
    return {"method": value} if isinstance(value, str) else value


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
    BeforeValidator(_expand_method_shorthand),
]


class LayerGroupInitialization(BaseModel):
    """Optional weight and bias overrides for one semantic layer group."""

    model_config = ConfigDict(extra="forbid")

    weight: InitializationMethodConfig | None = None
    bias: InitializationMethodConfig | None = None

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
            for parameter_kind in ("weight", "bias")
            if getattr(initialization, parameter_kind) is not None
        }


class InitializationMethodCandidates(BaseModel):
    """Candidate initialization methods for one weight or bias group."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[InitializationMethodConfig] = Field(..., min_length=1)


InitializationMethodSamplingConfig: TypeAlias = Union[
    InitializationMethodConfig,
    InitializationMethodCandidates,
]


class LayerGroupInitializationSampling(BaseModel):
    """Fixed or sampled weight and bias initialization for one layer group."""

    model_config = ConfigDict(extra="forbid")

    weight: InitializationMethodSamplingConfig | None = None
    bias: InitializationMethodSamplingConfig | None = None

    @model_validator(mode="after")
    @beartype
    def validate_not_empty(self):
        if self.weight is None and self.bias is None:
            raise ValueError("layer initialization must configure weight or bias")
        return self


class ModelInitializationSamplingConfig(
    RootModel[dict[LayerGroup, LayerGroupInitializationSampling]]
):
    """Hyperparameter sampling configuration for initialization overrides."""

    root: dict[LayerGroup, LayerGroupInitializationSampling] = Field(
        default_factory=dict
    )

    @staticmethod
    @beartype
    def _validation_method(
        value: InitializationMethodSamplingConfig,
    ) -> InitializationMethodConfig:
        if isinstance(value, InitializationMethodCandidates):
            return value.candidates[0].model_copy(deep=True)
        return value.model_copy(deep=True)

    @staticmethod
    @beartype
    def _sample_method(
        trial: Any,
        name: str,
        value: InitializationMethodSamplingConfig,
    ) -> InitializationMethodConfig:
        if not isinstance(value, InitializationMethodCandidates):
            return value.model_copy(deep=True)
        candidate_index = int(
            trial.suggest_categorical(
                name,
                list(range(len(value.candidates))),
            )
        )
        return value.candidates[candidate_index].model_copy(deep=True)

    @beartype
    def validation_config(self) -> ModelInitializationConfig:
        """Build a concrete config from the first value in every candidate list."""

        values: dict[LayerGroup, LayerGroupInitialization] = {}
        for group, sampling in self.root.items():
            values[group] = LayerGroupInitialization(
                weight=(
                    self._validation_method(sampling.weight)
                    if sampling.weight is not None
                    else None
                ),
                bias=(
                    self._validation_method(sampling.bias)
                    if sampling.bias is not None
                    else None
                ),
            )
        return ModelInitializationConfig(values)

    @beartype
    def sample_trial(
        self,
        trial: Any,
        parameter_prefix: str = "initialization",
    ) -> ModelInitializationConfig:
        """Sample candidate methods and return one concrete initialization config."""

        values: dict[LayerGroup, LayerGroupInitialization] = {}
        for group, sampling in self.root.items():
            group_prefix = f"{parameter_prefix}_{group.replace('.', '_')}"
            values[group] = LayerGroupInitialization(
                weight=(
                    self._sample_method(
                        trial,
                        f"{group_prefix}_weight_idx",
                        sampling.weight,
                    )
                    if sampling.weight is not None
                    else None
                ),
                bias=(
                    self._sample_method(
                        trial,
                        f"{group_prefix}_bias_idx",
                        sampling.bias,
                    )
                    if sampling.bias is not None
                    else None
                ),
            )
        return ModelInitializationConfig(values)

    @beartype
    def grid_size(self) -> int:
        """Return the cartesian count of configured initialization candidates."""

        candidate_counts = []
        for sampling in self.root.values():
            for value in (sampling.weight, sampling.bias):
                if isinstance(value, InitializationMethodCandidates):
                    candidate_counts.append(len(value.candidates))
        return math.prod(candidate_counts)
