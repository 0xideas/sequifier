"""Policy alignment and standard model-component configuration."""

import math
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from sequifier.config.metadata import FeatureTransforms, RawInputMetadata
from sequifier.config.naming import ConfigModel
from sequifier.io.values import DomainValue
from sequifier.model.execution_schema import ExecutionSchema
from sequifier.special_tokens import SPECIAL_TOKEN_IDS

NonemptyStr = Annotated[str, Field(min_length=1)]


class PolicyConfigBase(ConfigModel):
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

    @property
    def episode_horizon(self) -> int | None:
        return self.returns.finite_horizon if self.return_conditioned else None

    def categorical_id_maps(self) -> dict[str, dict[str | int, int]]:
        """Build target and feedback vocabularies in declared action order."""
        maps: dict[str, dict[str | int, int]] = {}
        for name, action in self.actions.items():
            if action.type == "categorical":
                if not action.values:
                    raise ValueError(
                        f"{name}: fresh categorical targets need values or metadata"
                    )
                mapping: dict[str | int, int] = {
                    str(label): index
                    for index, label in enumerate(
                        action.values, SPECIAL_TOKEN_IDS.user_start
                    )
                }
                if len(mapping) != len(action.values):
                    raise ValueError(
                        f"{name}: action labels must have distinct serialized keys"
                    )
                maps[name] = mapping

        feedback = {}
        for column, action_name in self.previous_action_inputs.items():
            if action_name in maps:
                mapping = maps[action_name].copy()
                initial = self.initial_inputs.get(column)
                if (
                    initial is not None
                    and initial not in self.actions[action_name].values
                ):
                    key = str(initial)
                    mapping.setdefault(
                        key,
                        SPECIAL_TOKEN_IDS.ids_by_label.get(
                            key, SPECIAL_TOKEN_IDS.user_start + len(mapping)
                        ),
                    )
                feedback[column] = mapping
        return maps | feedback

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
    """Executable view of a task with its model-owned input schema and transforms."""

    context_length: int = Field(gt=0)
    interface_name: NonemptyStr
    observation_columns: list[str] = Field(default_factory=list)
    static_columns: list[str] = Field(default_factory=list)
    action_order: list[str] = Field(default_factory=list)
    input_schema: ExecutionSchema | None = None
    transforms: FeatureTransforms = Field(default_factory=FeatureTransforms)
    raw_inputs: RawInputMetadata | None = None
    input_encoding: Literal["raw", "model"] = "raw"

    @field_validator("input_schema", mode="before")
    @classmethod
    def read_schema(cls, value):
        return ExecutionSchema.from_dict(value) if isinstance(value, dict) else value

    @property
    def task_binding(self) -> PolicyBinding:
        # These fields were validated together and belong to the same attached task.
        return PolicyBinding.model_construct(
            **{name: getattr(self, name) for name in PolicyBinding.model_fields}
        )

    @property
    def raw_mask_column(self):
        return self.raw_inputs.raw_mask_column if self.raw_inputs else None

    @property
    def id_maps(self):
        return self.transforms.id_maps

    @property
    def action_columns(self):
        return self.action_order or list(self.actions)

    @property
    def categorical_n_classes(self):
        return self.transforms.n_classes
