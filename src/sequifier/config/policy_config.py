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


class ActionFieldConfig(PolicyConfigBase):
    """Distribution and domain of one named model target."""

    type: Literal["categorical", "real"]
    values: list[DomainValue] = Field(default_factory=list)
    scale: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    learn_scale: bool = False
    min_scale: float = Field(default=1e-4, gt=0)
    max_scale: float = Field(default=100.0, gt=0)
    lower: float | None = Field(default=None, allow_inf_nan=False)
    upper: float | None = Field(default=None, allow_inf_nan=False)
    active_when: dict[str, list[DomainValue]] = Field(default_factory=dict)
    inactive_value: DomainValue = 0.0
    # Fixed point masses allow exact endpoints alongside a continuous density.
    atoms: dict[float, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_distribution(self):
        if self.type == "real" and (self.scale is None or self.values):
            raise ValueError("real actions require an explicit scale and no values")
        if self.type == "categorical" and (self.scale is not None or self.learn_scale):
            raise ValueError("categorical actions cannot configure a real scale")
        if len(set(self.values)) != len(self.values):
            raise ValueError("action labels must be unique")
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower >= self.upper
        ):
            raise ValueError("action lower bound must be below upper bound")
        if self.min_scale > self.max_scale:
            raise ValueError("min_scale must not exceed max_scale")
        if (
            self.scale is not None
            and not self.min_scale <= self.scale <= self.max_scale
        ):
            raise ValueError("scale must lie between min_scale and max_scale")
        if self.atoms and self.type != "real":
            raise ValueError("point-mass mixtures are real action distributions")
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


class PolicySchemaConfig(PolicyConfigBase):
    alignment: Literal["current_observation_previous_action", "model_inputs"] = (
        "current_observation_previous_action"
    )
    observation_columns: list[str] = Field(default_factory=list)
    static_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    # Numeric trajectory fields may also carry already encoded categorical IDs.
    observation_n_classes: dict[str, Annotated[int, Field(strict=True, ge=2)]] = Field(
        default_factory=dict
    )
    static_n_classes: dict[str, Annotated[int, Field(strict=True, ge=2)]] = Field(
        default_factory=dict
    )
    # The same label -> global ID convention as preprocessing metadata.
    id_maps: dict[str, dict[str | int, int]] = Field(default_factory=dict)
    categorical_encoding: Literal["global"] = "global"
    action_column: str = Field(default="action", min_length=1)
    action_values: list[DomainValue] = Field(
        default_factory=lambda: [0.0, 1.0], min_length=2
    )
    actions: dict[str, ActionFieldConfig] = Field(default_factory=dict)
    action_order: list[str] = Field(default_factory=list)
    decision_position: int = Field(default=-1, le=-1)
    # Existing input -> lagged action; no implicit features in model_inputs mode.
    previous_action_inputs: dict[str, str] = Field(default_factory=dict)
    initial_inputs: dict[str, Any] = Field(default_factory=dict)
    input_schema: dict[str, Any] | None = None
    input_statistics: dict[str, dict[str, float]] = Field(default_factory=dict)
    input_encoding: Literal["raw", "model"] = "raw"
    return_conditioned: bool = True
    initial_desired_return: float | None = Field(default=None, allow_inf_nan=False)
    returns: ReturnDefinition = Field(default_factory=ReturnDefinition)

    @model_validator(mode="after")
    def validate_names(self):
        if self.action_order and (
            set(self.action_order) != set(self.actions)
            or len(self.action_order) != len(self.actions)
        ):
            raise ValueError("action_order must name every action field exactly once")
        names = self.observation_columns + self.static_columns
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError(
                "policy observation/static columns must be nonempty and disjoint"
            )
        reserved = {"__policy_previous_action", "__policy_return", "__policy_action"}
        if self.alignment != "model_inputs" and reserved.intersection(
            names + [self.action_column]
        ):
            raise ValueError("policy feature names use reserved internal columns")
        for counts, columns in (
            (self.observation_n_classes, self.observation_columns),
            (self.static_n_classes, self.static_columns),
        ):
            if counts.keys() - set(columns) - set(self.previous_action_inputs):
                raise ValueError(
                    "class counts must name declared observation/static columns"
                )
        counts = {**self.observation_n_classes, **self.static_n_classes}
        if len(set(self.categorical_columns)) != len(self.categorical_columns) or set(
            self.categorical_columns
        ) - set(names) - set(self.previous_action_inputs):
            raise ValueError(
                "categorical_columns must name unique observation/static features"
            )
        if set(self.id_maps) - set(counts):
            raise ValueError("id_maps must name declared categorical inputs")
        for column, mapping in self.id_maps.items():
            if (
                not mapping
                or len(set(mapping.values())) != len(mapping)
                or any(
                    value < 0 or value >= counts[column] for value in mapping.values()
                )
            ):
                raise ValueError(f"{column}: invalid categorical ID map")
        if len(self.action_values) != len(set(self.action_values)):
            raise ValueError("policy action_values must be unique")
        if len({isinstance(value, str) for value in self.action_values}) != 1:
            raise ValueError(
                "action_values must use consistently numeric or string labels"
            )
        return self

    @property
    def action_columns(self) -> list[str]:
        return (
            (self.action_order or list(self.actions))
            if self.actions
            else [self.action_column]
        )

    @property
    def is_joint(self) -> bool:
        return bool(self.actions)

    @property
    def categorical_n_classes(self) -> dict[str, int]:
        return {**self.observation_n_classes, **self.static_n_classes}


class CategoricalPolicyConfig(PolicySchemaConfig):
    """Resolved alignment metadata; architecture belongs to the ordinary model."""

    context_length: int = Field(gt=0)
    interface_name: NonemptyStr


# General names for the shared contract; retain the original public names.
PolicyBinding = PolicySchemaConfig
PolicyConfig = CategoricalPolicyConfig
