"""Shared phase lifecycle declarations; algorithms only constrain capabilities."""

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import ConfigDict, Field, model_validator

from sequifier.config.naming import ConfigModel
from sequifier.config.sources import PhaseSources, SourceKind

if TYPE_CHECKING:
    from sequifier.config.policy_plan import OnlineEnvironmentConfig, ReferenceConfig


class EpochBudget(ConfigModel):
    epochs: int = Field(gt=0)


class IterationBudget(ConfigModel):
    iterations: int = Field(gt=0)


class Cadence(ConfigModel):
    epochs: int | None = Field(default=None, gt=0)
    iterations: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def one_clock(self):
        if (self.epochs is None) == (self.iterations is None):
            raise ValueError("cadence requires exactly one of epochs or iterations")
        return self


class EvaluationTriggers(ConfigModel):
    cadence: Cadence | None = None
    before_phase: bool = False
    on_snapshot: bool = False


class PhaseCheckpoints(ConfigModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    cadence: Cadence | None = None
    snapshot_interval_minutes: float | None = Field(default=None, gt=0)
    snapshot_interval_batches: int | None = Field(default=None, gt=0)


class ModelSelection(ConfigModel):
    handoff: Literal["best", "last"] = "last"
    scope: Literal["compatible_phases", "phase"] = "compatible_phases"


class StoppingConfig(ConfigModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    patience: int = Field(gt=0)
    min_delta: float = Field(default=0.0, ge=0)
    on_trigger: Literal["end_phase", "end_plan"] = "end_phase"


class CurriculumConfig(ConfigModel):
    column: str = Field(min_length=1)


class PhaseControls(ConfigModel):
    source_kind: ClassVar[SourceKind] = "sequence"

    @property
    def data_sources(self) -> PhaseSources:
        return PhaseSources.from_phase(self)

    if TYPE_CHECKING:
        algorithm: Any
        evaluation: Any

    checkpoints: PhaseCheckpoints = Field(default_factory=PhaseCheckpoints)
    model_selection: ModelSelection = Field(default_factory=ModelSelection)
    stopping: StoppingConfig | None = None

    @property
    def reference_config(self) -> "ReferenceConfig | None":
        return None

    @property
    def rollout_archive_directory(self) -> str | None:
        return None

    @property
    def training_environment(self) -> "OnlineEnvironmentConfig | None":
        return None

    @property
    def evaluation_environment(self) -> "OnlineEnvironmentConfig | None":
        return None

    @property
    def uses_actor_history(self) -> bool:
        return False

    @property
    def handoff(self):
        return self.model_selection.handoff

    @property
    def evaluation_interval_epochs(self):
        cadence = self.evaluation.cadence if self.evaluation is not None else None
        return cadence.epochs if cadence is not None else 1

    @property
    def checkpoint_interval_epochs(self):
        return self.checkpoints.cadence.epochs if self.checkpoints.cadence else 1

    @property
    def evaluation_interval_iterations(self):
        cadence = self.evaluation.cadence if self.evaluation is not None else None
        return cadence.iterations if cadence is not None else 1

    @property
    def checkpoint_interval_iterations(self):
        cadence = self.checkpoints.cadence
        return cadence.iterations if cadence else 1

    @model_validator(mode="after")
    def validate_lifecycle(self):
        if self.algorithm != "supervised" and self.model_selection.scope == "phase":
            raise ValueError("model_selection.scope is a supervised phase setting")
        clock = "epochs" if self.algorithm == "supervised" else "iterations"
        for label, cadence in (
            ("checkpoints", self.checkpoints.cadence),
            ("evaluation", self.evaluation.cadence if self.evaluation else None),
        ):
            if (
                cadence is not None
                and (cadence.epochs if clock == "epochs" else cadence.iterations)
                is None
            ):
                raise ValueError(f"{self.algorithm} {label}.cadence requires {clock}")
        if self.evaluation is None and (self.stopping or self.handoff == "best"):
            raise ValueError("stopping and best handoff require evaluation.monitor")
        if (
            self.stopping
            and self.evaluation is not None
            and self.evaluation.monitor is None
            and len(self.evaluation.sources or []) != 1
        ):
            raise ValueError("stopping requires evaluation.monitor")
        if self.algorithm != "supervised":
            if self.checkpoints.snapshot_interval_batches is not None:
                raise ValueError(
                    "policy snapshots use committed iterations, not batches"
                )
        return self
