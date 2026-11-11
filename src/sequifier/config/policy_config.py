"""Policy alignment and standard model-component configuration."""

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sequifier.io.values import DomainValue

NonemptyStr = Annotated[str, Field(min_length=1)]


class PolicyConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ReturnDefinition(PolicyConfigBase):
    gamma: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    finite_horizon: int | None = Field(default=None, gt=0)


class FixedActionScale(PolicyConfigBase):
    fixed: float = Field(gt=0)


class ActionFieldConfig(PolicyConfigBase):
    """Task-specific action support and selection of model or fixed scale."""

    type: Literal["categorical", "real"]
    values: list[DomainValue] = Field(default_factory=list)
    scale: Literal["model"] | FixedActionScale | None = None
    lower: float | None = None
    upper: float | None = None
    active_when: dict[str, list[DomainValue]] = Field(default_factory=dict)
    inactive_value: DomainValue = 0.0
    atoms: dict[float, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_distribution(self):
        if self.type == "real":
            if self.values:
                raise ValueError("real actions cannot declare categorical values")
            if self.scale is None:
                raise ValueError(
                    "real actions require scale: model or scale: {fixed: ...}"
                )
        elif (
            self.scale is not None
            or self.lower is not None
            or self.upper is not None
            or self.atoms
        ):
            raise ValueError(
                "categorical actions cannot configure real scale, bounds, or atoms"
            )
        if len(set(self.values)) != len(self.values):
            raise ValueError("action labels must be unique")
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower >= self.upper
        ):
            raise ValueError("action lower bound must be below upper bound")
        if (
            any(
                not math.isfinite(value)
                or not math.isfinite(probability)
                or probability <= 0
                for value, probability in self.atoms.items()
            )
            or sum(self.atoms.values()) >= 1
        ):
            raise ValueError(
                "atom probabilities must be positive and sum to less than one"
            )
        if any(
            (self.lower is not None and value < self.lower)
            or (self.upper is not None and value > self.upper)
            for value in self.atoms
        ):
            raise ValueError("point masses must lie within the declared action domain")
        return self


class PolicyBinding(PolicyConfigBase):
    """A task attached to named targets of an ordinary model interface."""

    actions: dict[str, ActionFieldConfig] = Field(min_length=1)
    decision_position: int = Field(default=-1, le=-1)
    previous_action_inputs: dict[str, str] = Field(default_factory=dict)
    initial_inputs: dict[str, Any] = Field(default_factory=dict)
    return_conditioned: bool = False
    initial_desired_return: float | None = Field(default=None, allow_inf_nan=False)
    returns: ReturnDefinition = Field(default_factory=ReturnDefinition)
    units: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_input_bindings(self):
        if any(not name for name in self.actions):
            raise ValueError("actions require nonempty target names")
        if set(self.previous_action_inputs.values()) - set(self.actions):
            raise ValueError("previous_action_inputs must reference declared actions")
        if set(self.initial_inputs) - set(self.previous_action_inputs):
            raise ValueError("initial_inputs must name previous_action_inputs")
        return self


class ResolvedPolicyBinding(PolicyBinding):
    """Fitted task binding persisted with a model's execution contract."""

    context_length: int = Field(gt=0)
    interface_name: NonemptyStr
    observation_columns: list[str] = Field(default_factory=list)
    static_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    observation_n_classes: dict[str, int] = Field(default_factory=dict)
    static_n_classes: dict[str, int] = Field(default_factory=dict)
    id_maps: dict[str, dict[str | int, int]] = Field(default_factory=dict)
    categorical_encoding: Literal["global"] = "global"
    action_order: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any]
    input_statistics: dict[str, dict[str, float]] = Field(default_factory=dict)
    input_encoding: Literal["raw", "model"] = "raw"

    @property
    def action_columns(self):
        return self.action_order or list(self.actions)

    @property
    def categorical_n_classes(self):
        return {**self.observation_n_classes, **self.static_n_classes}
