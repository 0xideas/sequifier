"""Shared authored phase contracts; algorithm-specific budgets remain explicit."""

import math
from typing import Annotated, Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sequifier.config.components import ComponentSpec
from sequifier.config.identifiers import identifier as _identifier
from sequifier.typechecking import beartype


class TrainingSourceSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    weight: Optional[float] = Field(default=None, gt=0)
    batches_per_selection: Optional[int] = Field(default=None, gt=0)

    @field_validator("source")
    @classmethod
    @beartype
    def validate_source(cls, value):
        parts = value.split(".")
        if len(parts) > 2:
            raise ValueError("Source refs use only dataset or dataset.part")
        for part in parts:
            _identifier(part, "Source reference component")
        return value


class TrainingPhaseSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["supervised"] = "supervised"

    name: str
    epochs: int = Field(gt=0)
    mode: Literal["sequential", "interleaved"]
    selection: Optional[Literal["round_robin", "weighted_random"]] = None
    sources: list[TrainingSourceSpecModel] = Field(..., min_length=1)

    @field_validator("name")
    @classmethod
    @beartype
    def validate_name(cls, value):
        return _identifier(value, "Training phase name")

    @model_validator(mode="after")
    @beartype
    def validate_mode_fields(self):
        if self.mode == "sequential":
            if self.selection is not None:
                raise ValueError("Sequential phases cannot configure selection")
            if any(source.batches_per_selection is not None for source in self.sources):
                raise ValueError(
                    "batches_per_selection is invalid in sequential phases"
                )
            if any(source.weight is not None for source in self.sources):
                raise ValueError("weight is invalid in sequential phases")
        else:
            self.selection = self.selection or "round_robin"
            if self.selection == "round_robin" and any(
                source.weight is not None for source in self.sources
            ):
                raise ValueError("weight is only valid for weighted_random selection")
        return self


class PolicyImitationPhaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    algorithm: Literal["policy_imitation"]
    name: str = Field(default="policy", min_length=1)
    epochs: int = Field(gt=0)
    sources: list[TrainingSourceSpecModel] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sources(self):
        if any(
            value.weight is not None or value.batches_per_selection is not None
            for value in self.sources
        ):
            raise ValueError(
                "policy sources do not support supervised source scheduling"
            )
        return self

    @property
    def trajectory_sources(self) -> list[str]:
        return [value.source for value in self.sources]

    validation_sources: list[str] | None = Field(default=None, min_length=1)
    policy: str = Field(min_length=1)


class IQLPhaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    algorithm: Literal["iql"]
    name: str = Field(default="offline_iql", min_length=1)
    policy: str = Field(min_length=1)
    sources: list[TrainingSourceSpecModel] = Field(min_length=1)
    validation_sources: list[str] | None = Field(default=None, min_length=1)
    updates: int = Field(gt=0)
    batch_size: int = Field(default=64, gt=0)
    gamma: float = Field(default=0.99, gt=0, le=1, allow_inf_nan=False)
    expectile: float = Field(default=0.7, gt=0, lt=1, allow_inf_nan=False)
    inverse_temperature: float = Field(default=3.0, gt=0, allow_inf_nan=False)
    max_weight: float = Field(default=100.0, ge=1, allow_inf_nan=False)
    target_update_rate: float = Field(default=0.005, gt=0, le=1, allow_inf_nan=False)
    actor_learning_rate: float = Field(default=3e-4, gt=0, allow_inf_nan=False)
    value_learning_rate: float = Field(default=3e-4, gt=0, allow_inf_nan=False)
    critic_learning_rate: float = Field(default=3e-4, gt=0, allow_inf_nan=False)
    actor_optimizer: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(name="Adam")
    )
    value_optimizer: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(name="Adam")
    )
    critic_optimizer: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(name="Adam")
    )
    actor_scheduler: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(
            name="ConstantLR", arguments={"factor": 1.0, "total_iters": 1}
        )
    )
    value_scheduler: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(
            name="ConstantLR", arguments={"factor": 1.0, "total_iters": 1}
        )
    )
    critic_scheduler: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(
            name="ConstantLR", arguments={"factor": 1.0, "total_iters": 1}
        )
    )
    gradient_clip: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    evaluation_interval: int = Field(default=100, gt=0)
    checkpoint_interval: int = Field(default=100, gt=0)

    @model_validator(mode="after")
    def validate_sources(self):
        if any(
            source.weight is not None or source.batches_per_selection is not None
            for source in self.sources
        ):
            raise ValueError("IQL sources use uniform transition sampling")
        return self

    @property
    def trajectory_sources(self) -> list[str]:
        return [value.source for value in self.sources]


class OnlineEnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    factory: str
    config: dict = Field(default_factory=dict)
    protocol_version: Literal[2] = 2
    # Empty means unitless; adapters must report this exact declaration.
    units: dict[str, str] = Field(default_factory=dict)

    @field_validator("factory")
    @classmethod
    def validate_factory(cls, value):
        module, separator, attribute = value.partition(":")
        if not separator or not module or not attribute.isidentifier():
            raise ValueError("environment.factory must use module:factory")
        return value


class PPONormalizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["identity", "statistics"]
    statistics: dict[str, dict[str, float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_statistics(self):
        if self.source == "identity" and self.statistics:
            raise ValueError("identity normalization cannot include statistics")
        for column, values in self.statistics.items():
            if (
                set(values) != {"mean", "std"}
                or not all(math.isfinite(value) for value in values.values())
                or values["std"] < 0
            ):
                raise ValueError(
                    f"{column}: statistics require finite mean and nonnegative std"
                )
        return self


class PPOEvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    environment: OnlineEnvironmentConfig
    seeds: list[int] = Field(min_length=1)
    max_episode_steps: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    selection: Literal["deterministic", "stochastic"] = "deterministic"
    metric: Literal[
        "mean_completed_return", "success_rate", "mean_completed_length"
    ] = "mean_completed_return"
    direction: Literal["min", "max"] = "max"

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, values):
        if any(value < 0 or value >= 2**63 for value in values):
            raise ValueError("evaluation seeds must be in [0, 2**63)")
        return values


class PPOResumeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy: Literal["never", "if_exists", "required"] = "never"
    mode: Literal["exact", "reset_environments"] = "reset_environments"


class PPOPhaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    algorithm: Literal["ppo"]
    name: str = Field(default="online_ppo", min_length=1)
    policy: str = Field(min_length=1)
    environment: OnlineEnvironmentConfig
    evaluation: PPOEvaluationConfig
    normalization: PPONormalizationConfig
    resume: PPOResumeConfig = Field(default_factory=PPOResumeConfig)
    seed_policy: Literal["derived_per_reset"] = "derived_per_reset"
    total_env_steps: int = Field(gt=0)
    rollout_length: int = Field(gt=0)
    allow_partial_rollout: bool = True
    optimization_epochs: int = Field(default=4, gt=0)
    minibatch_size: int = Field(default=64, gt=0)
    accumulation_steps: int = Field(default=1, gt=0)
    gamma: float = Field(default=0.99, ge=0, le=1, allow_inf_nan=False)
    gae_lambda: float = Field(default=0.95, ge=0, le=1, allow_inf_nan=False)
    clip_epsilon: float = Field(default=0.2, gt=0, lt=1, allow_inf_nan=False)
    value_clip: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    value_coefficient: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    entropy_coefficient: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    target_kl: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    normalize_advantages: bool = True
    reward_scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    actor_learning_rate: float = Field(default=3e-4, gt=0, allow_inf_nan=False)
    value_learning_rate: float = Field(default=3e-4, gt=0, allow_inf_nan=False)
    actor_optimizer: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(name="Adam")
    )
    value_optimizer: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(name="Adam")
    )
    actor_scheduler: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(
            name="ConstantLR", arguments={"factor": 1.0, "total_iters": 1}
        )
    )
    value_scheduler: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(
            name="ConstantLR", arguments={"factor": 1.0, "total_iters": 1}
        )
    )
    actor_gradient_clip: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    value_gradient_clip: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    evaluation_interval: int = Field(default=1, gt=0)
    checkpoint_interval: int = Field(default=1, gt=0)
    archive_directory: str | None = Field(default=None, min_length=1)
    archive_scores: bool = False
    max_buffer_bytes: int = Field(default=536870912, gt=0)

    @model_validator(mode="after")
    def validate_plans(self):
        if (
            not self.allow_partial_rollout
            and self.total_env_steps % self.rollout_length
        ):
            raise ValueError(
                "rollout_length must divide total_env_steps unless allow_partial_rollout"
            )
        for optimizer in (self.actor_optimizer, self.value_optimizer):
            if optimizer.name not in {"Adam", "AdamW", "SGD", "RMSprop"}:
                raise ValueError("PPO optimizers must be Adam, AdamW, SGD, or RMSprop")
            if "lr" in optimizer.arguments or "params" in optimizer.arguments:
                raise ValueError(
                    "PPO learning rates and parameter ownership belong to the phase"
                )
        full, remainder = divmod(self.total_env_steps, self.rollout_length)
        group_size = self.minibatch_size * self.accumulation_steps
        max_steps = self.optimization_epochs * (
            full * ((self.rollout_length + group_size - 1) // group_size)
            + ((remainder + group_size - 1) // group_size if remainder else 0)
        )
        for scheduler in (self.actor_scheduler, self.value_scheduler):
            if (
                scheduler.name == "OneCycleLR"
                and scheduler.arguments.get("total_steps") != max_steps
            ):
                raise ValueError(
                    "PPO OneCycleLR.total_steps must equal the maximum authored optimizer-step budget"
                )
            if (
                scheduler.name
                in {"ReduceLROnPlateau", "ChainedScheduler", "SequentialLR"}
                or "optimizer" in scheduler.arguments
            ):
                raise ValueError(
                    "PPO schedulers must advance independently on optimizer steps"
                )
        return self


Phase = TypeVar(
    "Phase",
    bound=TrainingPhaseSpecModel
    | PolicyImitationPhaseConfig
    | IQLPhaseConfig
    | PPOPhaseConfig,
)


class TrainingPlanModel(BaseModel, Generic[Phase]):
    model_config = ConfigDict(extra="forbid")

    curriculum_training: bool = False
    curriculum_column: Optional[str] = Field(default=None, min_length=1)
    phases: list[Phase] = Field(min_length=1)

    @field_validator("phases", mode="before")
    @classmethod
    def default_algorithm(cls, values):
        return [
            {"algorithm": "supervised", **value} if isinstance(value, dict) else value
            for value in values
        ]

    @model_validator(mode="after")
    @beartype
    def validate_unique_names(self):
        if self.curriculum_column is not None and not self.curriculum_training:
            raise ValueError(
                "curriculum_column may only be set when curriculum_training is true"
            )
        if self.curriculum_training and self.curriculum_column is None:
            raise ValueError(
                "curriculum_column must be set when curriculum_training is true"
            )
        if self.curriculum_training and any(
            phase.algorithm != "supervised" for phase in self.phases
        ):
            raise ValueError(
                "curriculum scheduling currently requires supervised phases"
            )
        names = [phase.name for phase in self.phases]
        if len(names) != len(set(names)):
            raise ValueError("Training phase names must be unique")
        return self


SupervisedTrainingPlanModel = TrainingPlanModel[TrainingPhaseSpecModel]
PolicyTrainingPlanModel = TrainingPlanModel[PolicyImitationPhaseConfig | IQLPhaseConfig]
PPOTrainingPlanModel = TrainingPlanModel[PPOPhaseConfig]
AlgorithmPhase = Annotated[
    TrainingPhaseSpecModel
    | PolicyImitationPhaseConfig
    | IQLPhaseConfig
    | PPOPhaseConfig,
    Field(discriminator="algorithm"),
]
