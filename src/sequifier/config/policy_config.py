"""Standalone, CPU-only categorical policy learning configuration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from sequifier.config.components import ComponentSpec


class PolicyConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReturnDefinition(PolicyConfigBase):
    gamma: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    finite_horizon: int | None = Field(default=None, gt=0)


class CategoricalPolicyConfig(PolicyConfigBase):
    alignment: Literal["current_observation_previous_action"] = (
        "current_observation_previous_action"
    )
    observation_columns: list[str] = Field(min_length=1)
    static_columns: list[str] = Field(default_factory=list)
    action_column: str = Field(default="action", min_length=1)
    action_values: list[FiniteFloat] = Field(
        default_factory=lambda: [0.0, 1.0], min_length=2
    )
    context_length: int = Field(default=4, gt=0)
    hidden_size: int = Field(default=32, gt=0)
    return_conditioned: bool = True
    returns: ReturnDefinition = Field(default_factory=ReturnDefinition)

    @model_validator(mode="after")
    def validate_names(self):
        names = self.observation_columns + self.static_columns
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError(
                "policy observation/static columns must be nonempty and disjoint"
            )
        if len(self.action_values) != len(set(self.action_values)):
            raise ValueError("policy action_values must be unique")
        return self


class PolicyTrainingConfig(PolicyConfigBase):
    epochs: int = Field(default=80, gt=0)
    batch_size: int = Field(default=64, gt=0)
    seed: int = Field(default=1010, ge=0)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    learning_rate: float = Field(default=0.005, gt=0, allow_inf_nan=False)
    gradient_clip: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    optimizer: ComponentSpec = Field(default_factory=lambda: ComponentSpec(name="Adam"))
    scheduler: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(
            name="ConstantLR", arguments={"factor": 1.0, "total_iters": 1}
        )
    )
    scheduler_step_on: Literal["epoch"] = "epoch"
    accumulation_steps: Literal[1] = 1
    layer_type_dtypes: None = None


class ControlExperimentConfig(PolicyConfigBase):
    project_root: str = "."
    output_dir: str = "outputs/numeric-policy"
    horizon: int = Field(default=3, ge=1, le=8)
    repetitions: int = Field(default=16, ge=2)
    context_length: int = Field(default=4, gt=0)
    hidden_size: int = Field(default=32, gt=0)
    returns: ReturnDefinition = Field(default_factory=ReturnDefinition)
    training: PolicyTrainingConfig = Field(default_factory=PolicyTrainingConfig)
    requested_returns: list[FiniteFloat] | None = None
    sampling_episodes: int = Field(default=8, gt=0)

    @model_validator(mode="after")
    def validate_horizon(self):
        if self.returns.finite_horizon not in (None, self.horizon):
            raise ValueError(
                "control experiment finite_horizon must match its task horizon"
            )
        if self.requested_returns == []:
            raise ValueError("requested_returns must be nonempty when supplied")
        return self
