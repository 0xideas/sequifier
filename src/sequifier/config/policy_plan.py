"""Built-in policy phase schemas; budgets and objectives are algorithm-specific."""

from typing import Annotated, ClassVar, Generic, Literal, TypeVar, get_args

from pydantic import (
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from sequifier.config.identifiers import identifier
from sequifier.config.naming import ConfigModel
from sequifier.config.normalization_config import (  # noqa: F401
    FrozenNormalizationConfig,
)
from sequifier.config.parameter_selection import TrainabilityConfig
from sequifier.config.phase_controls import (
    Cadence,
    EpochBudget,
    EvaluationTriggers,
    IterationBudget,
    PhaseCheckpoints,
    PhaseControls,
)
from sequifier.config.policy_optimization import OptimizationConfig
from sequifier.config.rollout_config import SequenceOutput
from sequifier.config.sources import TrainingSourceSpecModel


class OfflineEvaluationMonitor(ConfigModel):
    metric: Literal["validation_action_nll"] = "validation_action_nll"
    direction: Literal["min"] = "min"


class DPOEvaluationMonitor(ConfigModel):
    metric: Literal["validation_dpo_loss"] = "validation_dpo_loss"
    direction: Literal["min"] = "min"


class PolicyEvaluationTriggers(EvaluationTriggers):
    batch_size: int = Field(default=1, gt=0)


class OfflineEvaluationConfig(PolicyEvaluationTriggers):
    model_config = ConfigDict(extra="forbid")
    sources: list[TrainingSourceSpecModel] | None = Field(default=None, min_length=1)
    monitor: OfflineEvaluationMonitor = Field(default_factory=OfflineEvaluationMonitor)


class DPOEvaluationConfig(PolicyEvaluationTriggers):
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
    # Filled from the phases mapping key before validation.
    name: SkipJsonSchema[str]
    optimizer_state: Literal["reset", "continue"] = "reset"
    model_interface: str | None = None
    trainability: TrainabilityConfig = Field(default_factory=TrainabilityConfig)
    optimization_roles: ClassVar[tuple[str, ...]] = ("actor",)
    optimization: dict[str, OptimizationConfig] = Field(default_factory=dict)

    def optimizer_step_budget(self, training_size=None):
        raise NotImplementedError("built-in phase must declare its optimizer budget")

    def iteration_budget(self, optimizer_step_budget: int | None) -> int:
        return optimizer_step_budget or 1

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
        if self.uses_actor_history and self.optimizer_state == "continue":
            raise ValueError("history_initialization requires optimizer_state: reset")
        self.trainability.validate_roles(self.optimization_roles)
        for config in self.optimization_configs().values():
            config.validate_budget(self.optimizer_step_budget())
        return self

    @model_serializer(mode="wrap")
    def serialize_history_initialization(self, handler):
        values = handler(self)
        if values.get("history_initialization") == "random":
            values.pop("history_initialization")
        return values

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
    budget: EpochBudget

    def optimizer_step_budget(self, training_size=None):
        if training_size is None:
            return None
        size = self.batch_size * self.accumulation_batches
        return self.budget.epochs * ((training_size + size - 1) // size)


class IQLPhaseConfig(OfflinePhaseConfig):
    optimization_roles: ClassVar[tuple[str, ...]] = ("actor", "value", "critic")
    algorithm: Literal["iql"]
    history_initialization: Literal["random", "phase_start_actor"] = "random"
    evaluation: OfflineEvaluationConfig = Field(
        default_factory=lambda: OfflineEvaluationConfig(cadence=Cadence(iterations=100))
    )
    checkpoints: PhaseCheckpoints = Field(
        default_factory=lambda: PhaseCheckpoints(cadence=Cadence(iterations=100))
    )
    budget: IterationBudget
    batch_size: int = Field(default=64, gt=0)
    gamma: float = Field(default=0.99, gt=0, le=1, allow_inf_nan=False)
    expectile: float = Field(default=0.7, gt=0, lt=1, allow_inf_nan=False)
    inverse_temperature: float = Field(default=3.0, gt=0, allow_inf_nan=False)
    max_advantage_weight: float = Field(default=100.0, ge=1, allow_inf_nan=False)
    target_update_rate: float = Field(default=0.005, gt=0, le=1, allow_inf_nan=False)

    @property
    def uses_actor_history(self) -> bool:
        return self.history_initialization == "phase_start_actor"

    def iteration_budget(self, optimizer_step_budget: int | None) -> int:
        return self.budget.iterations

    def optimizer_step_budget(self, training_size=None):
        return self.budget.iterations


class OnlineEnvironmentConfig(ConfigModel):
    factory: str
    config: dict = Field(default_factory=dict)

    # Empty means unitless; adapters must report this exact declaration.
    units: dict[str, str] = Field(default_factory=dict)

    @field_validator("factory")
    @classmethod
    def validate_factory(cls, value):
        module, separator, attribute = value.partition(":")
        if not separator or not module or not attribute.isidentifier():
            raise ValueError("environment.factory must use module:factory")
        return value


class OnlineCollection(ConfigModel):
    output: SequenceOutput | None = None
    environment: str | SkipJsonSchema[OnlineEnvironmentConfig]
    archive_directory: str | None = Field(default=None, min_length=1)
    max_buffer_bytes: int = Field(default=536870912, gt=0)


class PPOCollection(OnlineCollection):
    env_steps_per_iteration: int = Field(gt=0)
    allow_partial_rollout: bool = True
    archive_scores: bool = False

    @model_validator(mode="after")
    def require_archive(self):
        if self.archive_scores and self.archive_directory is None:
            raise ValueError("collection.archive_scores requires an archive_directory")
        return self


class PPOBudget(ConfigModel):
    env_steps: int = Field(gt=0)


class GroupBudget(ConfigModel):
    groups: int = Field(gt=0)
    max_env_steps: int | None = Field(default=None, gt=0)


class ParallelCollection(ConfigModel):
    workers: int = Field(ge=2)
    inference_batch_size: int = Field(default=32, gt=0)
    timeout_seconds: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    threads_per_worker: int = Field(default=1, gt=0)


class GroupCollection(OnlineCollection):
    groups_per_iteration: int = Field(default=1, gt=0)
    group_size: int = Field(default=4, ge=2)
    max_steps: int = Field(gt=0)
    parallel: ParallelCollection | None = None
    image_feature_cache_bytes: int = Field(default=0, ge=0)


class EvaluationEpisodesConfig(ConfigModel):
    environment: str | SkipJsonSchema[OnlineEnvironmentConfig]
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
    metric: Literal[
        "mean_completed_return", "success_rate", "mean_completed_length", "score_mean"
    ] = "mean_completed_return"
    direction: Literal["min", "max"] = "max"


class OnlineEvaluationConfig(EvaluationEpisodesConfig, PolicyEvaluationTriggers):
    @model_validator(mode="after")
    def serial_evaluation(self):
        if self.batch_size != 1:
            raise ValueError(
                "online evaluation runs serial episodes; evaluation.batch_size must be 1"
            )
        return self

    monitor: OnlineEvaluationMonitor = Field(default_factory=OnlineEvaluationMonitor)


CollectionT = TypeVar("CollectionT", bound=OnlineCollection)
EvaluationT = TypeVar("EvaluationT", bound=OnlineEvaluationConfig)


class OnlinePhaseConfig(PhaseConfig, Generic[CollectionT, EvaluationT]):
    source_kind = "none"
    collection: CollectionT
    evaluation: EvaluationT

    @property
    def rollout_archive_directory(self) -> str | None:
        return self.collection.archive_directory

    @property
    def training_environment(self):
        environment = self.collection.environment
        return environment if isinstance(environment, OnlineEnvironmentConfig) else None

    @property
    def evaluation_environment(self):
        environment = self.evaluation.environment
        return environment if isinstance(environment, OnlineEnvironmentConfig) else None


def _ceil_div(total: int, size: int) -> int:
    return (total + size - 1) // size


def _collection_optimizer_steps(phase, total, per_iteration, samples_per_unit=1):
    full, remainder = divmod(total, per_iteration)
    batch = phase.batch_size * phase.accumulation_batches
    return phase.optimization_epochs * (
        full * _ceil_div(per_iteration * samples_per_unit, batch)
        + _ceil_div(remainder * samples_per_unit, batch)
    )


class PPOPhaseConfig(OnlinePhaseConfig[PPOCollection, OnlineEvaluationConfig]):
    optimization_roles: ClassVar[tuple[str, ...]] = ("actor", "value")
    algorithm: Literal["ppo"]
    history_initialization: Literal["random", "phase_start_actor"] = "random"
    evaluation: OnlineEvaluationConfig
    budget: PPOBudget
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

    @property
    def uses_actor_history(self) -> bool:
        return self.history_initialization == "phase_start_actor"

    def iteration_budget(self, optimizer_step_budget: int | None) -> int:
        return _ceil_div(self.budget.env_steps, self.collection.env_steps_per_iteration)

    def optimizer_step_budget(self, training_size=None):
        return _collection_optimizer_steps(
            self, self.budget.env_steps, self.collection.env_steps_per_iteration
        )

    @model_validator(mode="after")
    def validate_plans(self):
        if (
            not self.collection.allow_partial_rollout
            and self.budget.env_steps % self.collection.env_steps_per_iteration
        ):
            raise ValueError(
                "collection.env_steps_per_iteration must divide budget.env_steps unless collection.allow_partial_rollout is enabled"
            )
        return self


class ReferenceConfig(ConfigModel):
    source: Literal["phase_start", "artifact"] = "phase_start"
    path: str | None = None

    @model_validator(mode="after")
    def location(self):
        if (self.source == "artifact") != (self.path is not None):
            raise ValueError("reference.path is required only for source: artifact")
        return self


class DPOPhaseConfig(BatchedPhaseConfig):
    source_kind = "preference"
    algorithm: Literal["dpo"]
    sources: list[TrainingSourceSpecModel] = Field(min_length=1)
    evaluation: DPOEvaluationConfig = Field(default_factory=DPOEvaluationConfig)
    budget: IterationBudget
    beta: float = Field(default=0.1, gt=0)
    scoring_chunk_size: int = Field(default=64, gt=0)
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)

    @property
    def reference_config(self) -> ReferenceConfig:
        return self.reference

    def iteration_budget(self, optimizer_step_budget: int | None) -> int:
        return self.budget.iterations

    def optimizer_step_budget(self, training_size=None):
        return self.budget.iterations


class GRPOEvaluationConfig(OnlineEvaluationConfig):
    monitor: OnlineEvaluationMonitor = Field(
        default_factory=lambda: OnlineEvaluationMonitor(metric="score_mean")
    )


class GRPOPhaseConfig(
    BatchedPhaseConfig, OnlinePhaseConfig[GroupCollection, GRPOEvaluationConfig]
):
    scoring_chunk_size: int = Field(default=64, gt=0)
    budget: GroupBudget
    algorithm: Literal["grpo"]
    evaluation: GRPOEvaluationConfig
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    optimization_epochs: int = Field(default=4, gt=0)
    clip_epsilon: float = Field(default=0.2, gt=0, lt=1)
    advantage_epsilon: float = Field(default=1e-6, gt=0)
    reference_kl_coefficient: float = Field(default=0.04, ge=0)
    old_policy_target_kl: float | None = Field(default=None, gt=0)
    reward_scale: float = Field(default=1.0, gt=0)

    @property
    def reference_config(self) -> ReferenceConfig:
        return self.reference

    def iteration_budget(self, optimizer_step_budget: int | None) -> int:
        return _ceil_div(self.budget.groups, self.collection.groups_per_iteration)

    def optimizer_step_budget(self, training_size=None):
        return _collection_optimizer_steps(
            self,
            self.budget.groups,
            self.collection.groups_per_iteration,
            self.collection.group_size,
        )


PolicyPhaseVariant = (
    PolicyImitationPhaseConfig
    | IQLPhaseConfig
    | PPOPhaseConfig
    | DPOPhaseConfig
    | GRPOPhaseConfig
)
PolicyPhase = Annotated[PolicyPhaseVariant, Field(discriminator="algorithm")]

POLICY_PHASE_SCHEMAS = {
    algorithm: schema
    for schema in get_args(PolicyPhaseVariant)
    for algorithm in get_args(schema.model_fields["algorithm"].annotation)
}
