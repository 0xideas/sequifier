"""Built-in policy phase schemas; budgets and objectives are algorithm-specific."""

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sequifier.config.components import ComponentSpec
from sequifier.config.identifiers import identifier
from sequifier.config.policy_config import CategoricalPolicyConfig
from sequifier.config.policy_optimization import (
    OptimizationPlan,
    default_optimizer,
    default_scheduler,
)


class PhaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str
    handoff: Literal["best", "last"] = "best"
    representation: CategoricalPolicyConfig | None = None

    def optimizer_step_budget(self, training_size=None):
        raise NotImplementedError("built-in phase must declare its optimizer budget")

    def optimization_plans(self, training_size=None):
        raise NotImplementedError("built-in phase must declare optimization roles")

    @model_validator(mode="after")
    def validate_optimization(self):
        self.optimization_plans()
        return self

    @field_validator("name")
    @classmethod
    def phase_name(cls, value):
        return identifier(value, "Policy phase name")


class ActorOptimizationConfig(PhaseConfig):
    batch_size: int = Field(default=32, gt=0)
    accumulation_steps: int = Field(default=1, gt=0)
    actor_learning_rate: float = Field(default=3e-4, gt=0)
    actor_optimizer: ComponentSpec = Field(default_factory=default_optimizer)
    actor_scheduler: ComponentSpec = Field(default_factory=default_scheduler)
    gradient_clip: float = Field(default=1.0, gt=0)
    evaluation_interval: int = Field(default=1, gt=0)
    checkpoint_interval: int = Field(default=1, gt=0)

    def optimization_plans(self, training_size=None):
        budget = self.optimizer_step_budget(training_size)
        return {
            "actor": OptimizationPlan(
                self.actor_optimizer,
                self.actor_scheduler,
                self.actor_learning_rate,
                self.gradient_clip,
                budget,
            )
        }


class OfflinePhaseConfig(PhaseConfig):
    sources: list[str] = Field(min_length=1)
    validation_sources: list[str] | None = Field(default=None, min_length=1)


class PolicyImitationPhaseConfig(ActorOptimizationConfig, OfflinePhaseConfig):
    algorithm: Literal["policy_imitation"]
    epochs: int = Field(gt=0)
    monitor: Literal["validation_action_nll"] = "validation_action_nll"
    direction: Literal["min"] = "min"

    def optimizer_step_budget(self, training_size=None):
        if training_size is None:
            return None
        size = self.batch_size * self.accumulation_steps
        return self.epochs * ((training_size + size - 1) // size)


class IQLPhaseConfig(OfflinePhaseConfig):
    model_config = ConfigDict(extra="forbid")
    monitor: Literal["validation_action_nll"] = "validation_action_nll"
    direction: Literal["min"] = "min"
    algorithm: Literal["iql"]
    name: str = Field(default="offline_iql", min_length=1)
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
    actor_optimizer: ComponentSpec = Field(default_factory=default_optimizer)
    value_optimizer: ComponentSpec = Field(default_factory=default_optimizer)
    critic_optimizer: ComponentSpec = Field(default_factory=default_optimizer)
    actor_scheduler: ComponentSpec = Field(default_factory=default_scheduler)
    value_scheduler: ComponentSpec = Field(default_factory=default_scheduler)
    critic_scheduler: ComponentSpec = Field(default_factory=default_scheduler)
    gradient_clip: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    evaluation_interval: int = Field(default=100, gt=0)
    checkpoint_interval: int = Field(default=100, gt=0)

    def optimizer_step_budget(self, training_size=None):
        return self.updates

    def optimization_plans(self, training_size=None):
        budget = self.optimizer_step_budget(training_size)
        return {
            "actor": OptimizationPlan(
                self.actor_optimizer,
                self.actor_scheduler,
                self.actor_learning_rate,
                self.gradient_clip,
                budget,
            ),
            "value": OptimizationPlan(
                self.value_optimizer,
                self.value_scheduler,
                self.value_learning_rate,
                self.gradient_clip,
                budget,
            ),
            "critic": OptimizationPlan(
                self.critic_optimizer,
                self.critic_scheduler,
                self.critic_learning_rate,
                self.gradient_clip,
                budget,
            ),
        }


class OnlineEnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    factory: str
    config: dict = Field(default_factory=dict)
    protocol_version: Literal[3] = 3
    implementation_id: str = Field(min_length=1)
    # Empty means unitless; adapters must report this exact declaration.
    units: dict[str, str] = Field(default_factory=dict)

    @field_validator("factory")
    @classmethod
    def validate_factory(cls, value):
        module, separator, attribute = value.partition(":")
        if not separator or not module or not attribute.isidentifier():
            raise ValueError("environment.factory must use module:factory")
        return value


class FrozenNormalizationConfig(BaseModel):
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


class EvaluationEpisodesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    environment: OnlineEnvironmentConfig
    seeds: list[int] = Field(min_length=1)
    max_episode_steps: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    selection: Literal["deterministic", "stochastic"] = "deterministic"

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, values):
        if any(value < 0 or value >= 2**63 for value in values):
            raise ValueError("evaluation seeds must be in [0, 2**63)")
        return values


class OnlineEvaluationConfig(EvaluationEpisodesConfig):
    metric: Literal[
        "mean_completed_return", "success_rate", "mean_completed_length", "score_mean"
    ] = "mean_completed_return"
    direction: Literal["min", "max"] = "max"


class PolicyResumeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy: Literal["never", "if_exists", "required"] = "never"
    mode: Literal["exact", "reset_environments"] = "reset_environments"


class PPOPhaseConfig(PhaseConfig):
    model_config = ConfigDict(extra="forbid")
    algorithm: Literal["ppo"]
    name: str = Field(default="online_ppo", min_length=1)
    environment: OnlineEnvironmentConfig
    evaluation: OnlineEvaluationConfig
    normalization: FrozenNormalizationConfig | None = None
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
    actor_optimizer: ComponentSpec = Field(default_factory=default_optimizer)
    value_optimizer: ComponentSpec = Field(default_factory=default_optimizer)
    actor_scheduler: ComponentSpec = Field(default_factory=default_scheduler)
    value_scheduler: ComponentSpec = Field(default_factory=default_scheduler)
    actor_gradient_clip: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    value_gradient_clip: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    evaluation_interval: int = Field(default=1, gt=0)
    checkpoint_interval: int = Field(default=1, gt=0)
    archive_directory: str | None = Field(default=None, min_length=1)
    archive_scores: bool = False
    max_buffer_bytes: int = Field(default=536870912, gt=0)

    def optimizer_step_budget(self, training_size=None):
        full, remainder = divmod(self.total_env_steps, self.rollout_length)
        size = self.minibatch_size * self.accumulation_steps
        return self.optimization_epochs * (
            full * ((self.rollout_length + size - 1) // size)
            + ((remainder + size - 1) // size if remainder else 0)
        )

    @model_validator(mode="after")
    def validate_plans(self):
        if (
            not self.allow_partial_rollout
            and self.total_env_steps % self.rollout_length
        ):
            raise ValueError(
                "rollout_length must divide total_env_steps unless allow_partial_rollout"
            )
        return self

    def optimization_plans(self, training_size=None):
        budget = self.optimizer_step_budget(training_size)
        return {
            "actor": OptimizationPlan(
                self.actor_optimizer,
                self.actor_scheduler,
                self.actor_learning_rate,
                self.actor_gradient_clip,
                budget,
            ),
            "value": OptimizationPlan(
                self.value_optimizer,
                self.value_scheduler,
                self.value_learning_rate,
                self.value_gradient_clip,
                budget,
            ),
        }


class ReferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["phase_start", "artifact"] = "phase_start"
    path: str | None = None

    @model_validator(mode="after")
    def location(self):
        if (self.source == "artifact") != (self.path is not None):
            raise ValueError("reference.path is required only for source: artifact")
        return self


class DPOPhaseConfig(ActorOptimizationConfig):
    algorithm: Literal["dpo"]
    preferences: str
    validation_preferences: str | None = None
    updates: int = Field(gt=0)
    beta: float = Field(default=0.1, gt=0)
    scoring_chunk_size: int = Field(default=64, gt=0)
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    monitor: Literal["validation_dpo_loss"] = "validation_dpo_loss"
    direction: Literal["min"] = "min"

    def optimizer_step_budget(self, training_size=None):
        return self.updates


class GRPOEvaluationConfig(EvaluationEpisodesConfig):
    metric: Literal["score_mean"] = "score_mean"
    direction: Literal["max"] = "max"


class GRPOPhaseConfig(ActorOptimizationConfig):
    scoring_chunk_size: int = Field(default=64, gt=0)
    algorithm: Literal["grpo"]
    environment: OnlineEnvironmentConfig
    evaluation: GRPOEvaluationConfig
    normalization: FrozenNormalizationConfig | None = None
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    total_groups: int = Field(gt=0)
    groups_per_update: int = Field(default=1, gt=0)
    group_size: int = Field(default=4, ge=2)
    max_steps: int = Field(gt=0)
    max_env_steps: int | None = Field(default=None, gt=0)
    max_buffer_bytes: int = Field(default=536870912, gt=0)
    optimization_epochs: int = Field(default=4, gt=0)
    clip_epsilon: float = Field(default=0.2, gt=0, lt=1)
    advantage_epsilon: float = Field(default=1e-6, gt=0)
    kl_coefficient: float = Field(default=0.04, ge=0)
    old_policy_target_kl: float | None = Field(default=None, gt=0)
    reward_scale: float = Field(default=1.0, gt=0)
    archive_directory: str | None = None

    def optimizer_step_budget(self, training_size=None):
        full, remainder = divmod(self.total_groups, self.groups_per_update)
        size = self.batch_size * self.accumulation_steps
        return self.optimization_epochs * (
            full * ((self.groups_per_update * self.group_size + size - 1) // size)
            + ((remainder * self.group_size + size - 1) // size if remainder else 0)
        )


PolicyPhase = Annotated[
    PolicyImitationPhaseConfig
    | IQLPhaseConfig
    | PPOPhaseConfig
    | DPOPhaseConfig
    | GRPOPhaseConfig,
    Field(discriminator="algorithm"),
]
