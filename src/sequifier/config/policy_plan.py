"""Built-in policy phase schemas; budgets and objectives are algorithm-specific."""

from typing import Annotated, ClassVar, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from sequifier.config.identifiers import identifier
from sequifier.config.naming import ConfigModel
from sequifier.config.normalization_config import (  # noqa: F401
    FrozenNormalizationConfig,
)
from sequifier.config.parameter_selection import TrainabilityConfig
from sequifier.config.phase_controls import (
    Cadence,
    EvaluationTriggers,
    PhaseCheckpoints,
    PhaseControls,
)
from sequifier.config.policy_optimization import OptimizationConfig
from sequifier.config.sources import TrainingSourceSpecModel


class OfflineEvaluationMonitor(ConfigModel):
    model_config = ConfigDict(extra="forbid")
    metric: Literal["validation_action_nll"] = "validation_action_nll"
    direction: Literal["min"] = "min"


class DPOEvaluationMonitor(ConfigModel):
    model_config = ConfigDict(extra="forbid")
    metric: Literal["validation_dpo_loss"] = "validation_dpo_loss"
    direction: Literal["min"] = "min"


class OfflineEvaluationConfig(EvaluationTriggers):
    model_config = ConfigDict(extra="forbid")
    sources: list[TrainingSourceSpecModel] | None = Field(default=None, min_length=1)
    monitor: OfflineEvaluationMonitor = Field(default_factory=OfflineEvaluationMonitor)


class DPOEvaluationConfig(EvaluationTriggers):
    @model_validator(mode="after")
    def serial_evaluation(self):
        if self.batch_size != 1:
            raise ValueError(
                "DPO evaluation scores one preference pair at a time; evaluation.batch_size must be 1"
            )
        return self

    model_config = ConfigDict(extra="forbid")
    sources: list[TrainingSourceSpecModel] | None = Field(default=None, min_length=1)
    monitor: DPOEvaluationMonitor = Field(default_factory=DPOEvaluationMonitor)


class PhaseConfig(PhaseControls):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str
    optimizer_state: Literal["reset", "continue"] = "reset"
    model_interface: str | None = None
    trainability: TrainabilityConfig = Field(default_factory=TrainabilityConfig)
    optimization_roles: ClassVar[tuple[str, ...]] = ("actor",)
    optimization: dict[str, OptimizationConfig] = Field(default_factory=dict)

    def optimizer_step_budget(self, training_size=None):
        raise NotImplementedError("built-in phase must declare its optimizer budget")

    def optimization_configs(self):
        unknown = self.optimization.keys() - set(self.optimization_roles)
        if unknown:
            raise ValueError(f"unsupported optimization roles: {sorted(unknown)}")
        return {
            role: self.optimization.get(role, OptimizationConfig())
            for role in self.optimization_roles
        }

    @model_validator(mode="after")
    def validate_optimization(self):
        self.trainability.validate_roles(self.optimization_roles)
        for config in self.optimization_configs().values():
            config.validate_budget(self.optimizer_step_budget())
        return self

    @field_validator("name")
    @classmethod
    def phase_name(cls, value):
        return identifier(value, "Policy phase name")


class BatchedPhaseConfig(PhaseConfig):
    batch_size: int = Field(default=32, gt=0)
    accumulation_batches: int = Field(default=1, gt=0)


class OfflinePhaseConfig(PhaseConfig):
    sources: list[TrainingSourceSpecModel] = Field(min_length=1)
    evaluation: OfflineEvaluationConfig = Field(default_factory=OfflineEvaluationConfig)


class PolicyImitationPhaseConfig(BatchedPhaseConfig, OfflinePhaseConfig):
    algorithm: Literal["policy_imitation"]
    epochs: int = Field(gt=0)

    def optimizer_step_budget(self, training_size=None):
        if training_size is None:
            return None
        size = self.batch_size * self.accumulation_batches
        return self.epochs * ((training_size + size - 1) // size)


class IQLPhaseConfig(OfflinePhaseConfig):
    optimization_roles: ClassVar[tuple[str, ...]] = ("actor", "value", "critic")
    algorithm: Literal["iql"]
    evaluation: OfflineEvaluationConfig = Field(
        default_factory=lambda: OfflineEvaluationConfig(cadence=Cadence(iterations=100))
    )
    checkpoints: PhaseCheckpoints = Field(
        default_factory=lambda: PhaseCheckpoints(cadence=Cadence(iterations=100))
    )
    name: str = Field(default="offline_iql", min_length=1)
    iterations: int = Field(gt=0)
    batch_size: int = Field(default=64, gt=0)
    gamma: float = Field(default=0.99, gt=0, le=1, allow_inf_nan=False)
    expectile: float = Field(default=0.7, gt=0, lt=1, allow_inf_nan=False)
    inverse_temperature: float = Field(default=3.0, gt=0, allow_inf_nan=False)
    max_advantage_weight: float = Field(default=100.0, ge=1, allow_inf_nan=False)
    target_update_rate: float = Field(default=0.005, gt=0, le=1, allow_inf_nan=False)

    def optimizer_step_budget(self, training_size=None):
        return self.iterations


class OnlineEnvironmentConfig(ConfigModel):
    model_config = ConfigDict(extra="forbid")
    factory: str
    config: dict = Field(default_factory=dict)

    @property
    def protocol_version(self):
        return 3

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


class EvaluationEpisodesConfig(ConfigModel):
    model_config = ConfigDict(extra="forbid")
    environment: OnlineEnvironmentConfig
    seeds: list[int] = Field(min_length=1)
    max_episode_steps: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    action_selection: Literal["deterministic", "stochastic"] = "deterministic"

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, values):
        if any(value < 0 or value >= 2**63 for value in values):
            raise ValueError("evaluation seeds must be in [0, 2**63)")
        return values


class OnlineEvaluationMonitor(ConfigModel):
    model_config = ConfigDict(extra="forbid")

    metric: Literal[
        "mean_completed_return", "success_rate", "mean_completed_length", "score_mean"
    ] = "mean_completed_return"
    direction: Literal["min", "max"] = "max"


class OnlineEvaluationConfig(EvaluationEpisodesConfig, EvaluationTriggers):
    @model_validator(mode="after")
    def serial_evaluation(self):
        if self.batch_size != 1:
            raise ValueError(
                "online evaluation runs serial episodes; evaluation.batch_size must be 1"
            )
        return self

    monitor: OnlineEvaluationMonitor = Field(default_factory=OnlineEvaluationMonitor)


class PPOPhaseConfig(PhaseConfig):
    optimization_roles: ClassVar[tuple[str, ...]] = ("actor", "value")
    algorithm: Literal["ppo"]
    name: str = Field(default="online_ppo", min_length=1)
    environment: OnlineEnvironmentConfig
    evaluation: OnlineEvaluationConfig
    seed_policy: Literal["derived_per_reset"] = "derived_per_reset"
    total_env_steps: int = Field(gt=0)
    rollout_env_steps: int = Field(gt=0)
    allow_partial_rollout: bool = True
    optimization_epochs: int = Field(default=4, gt=0)
    batch_size: int = Field(default=64, gt=0)
    accumulation_batches: int = Field(default=1, gt=0)
    gamma: float = Field(default=0.99, ge=0, le=1, allow_inf_nan=False)
    gae_lambda: float = Field(default=0.95, ge=0, le=1, allow_inf_nan=False)
    clip_epsilon: float = Field(default=0.2, gt=0, lt=1, allow_inf_nan=False)
    value_clip_delta: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    value_coefficient: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    entropy_coefficient: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    old_policy_target_kl: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    normalize_advantages: bool = True
    reward_scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    archive_directory: str | None = Field(default=None, min_length=1)
    archive_scores: bool = False
    max_buffer_bytes: int = Field(default=536870912, gt=0)

    def optimizer_step_budget(self, training_size=None):
        full, remainder = divmod(self.total_env_steps, self.rollout_env_steps)
        size = self.batch_size * self.accumulation_batches
        return self.optimization_epochs * (
            full * ((self.rollout_env_steps + size - 1) // size)
            + ((remainder + size - 1) // size if remainder else 0)
        )

    @model_validator(mode="after")
    def validate_plans(self):
        if (
            not self.allow_partial_rollout
            and self.total_env_steps % self.rollout_env_steps
        ):
            raise ValueError(
                "rollout_env_steps must divide total_env_steps unless allow_partial_rollout"
            )
        return self


class ReferenceConfig(ConfigModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["phase_start", "artifact"] = "phase_start"
    path: str | None = None

    @model_validator(mode="after")
    def location(self):
        if (self.source == "artifact") != (self.path is not None):
            raise ValueError("reference.path is required only for source: artifact")
        return self


class DPOPhaseConfig(BatchedPhaseConfig):
    algorithm: Literal["dpo"]
    preferences: TrainingSourceSpecModel
    evaluation: DPOEvaluationConfig = Field(default_factory=DPOEvaluationConfig)
    iterations: int = Field(gt=0)
    beta: float = Field(default=0.1, gt=0)
    scoring_chunk_size: int = Field(default=64, gt=0)
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)

    def optimizer_step_budget(self, training_size=None):
        return self.iterations


class GRPOEvaluationConfig(OnlineEvaluationConfig):
    monitor: OnlineEvaluationMonitor = Field(
        default_factory=lambda: OnlineEvaluationMonitor(metric="score_mean")
    )


class GRPOPhaseConfig(BatchedPhaseConfig):
    scoring_chunk_size: int = Field(default=64, gt=0)
    algorithm: Literal["grpo"]
    environment: OnlineEnvironmentConfig
    evaluation: GRPOEvaluationConfig
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    total_groups: int = Field(gt=0)
    groups_per_iteration: int = Field(default=1, gt=0)
    group_size: int = Field(default=4, ge=2)
    max_steps: int = Field(gt=0)
    max_env_steps: int | None = Field(default=None, gt=0)
    max_buffer_bytes: int = Field(default=536870912, gt=0)
    optimization_epochs: int = Field(default=4, gt=0)
    clip_epsilon: float = Field(default=0.2, gt=0, lt=1)
    advantage_epsilon: float = Field(default=1e-6, gt=0)
    reference_kl_coefficient: float = Field(default=0.04, ge=0)
    old_policy_target_kl: float | None = Field(default=None, gt=0)
    reward_scale: float = Field(default=1.0, gt=0)
    archive_directory: str | None = None

    def optimizer_step_budget(self, training_size=None):
        full, remainder = divmod(self.total_groups, self.groups_per_iteration)
        size = self.batch_size * self.accumulation_batches
        return self.optimization_epochs * (
            full * ((self.groups_per_iteration * self.group_size + size - 1) // size)
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
