"""Read-only phase views for native supervised execution."""

from dataclasses import dataclass
from typing import Literal

from pydantic import ConfigDict, Field

from sequifier.config.train_config import ResolvedSequifierConfig, ResolvedTrainingPhase


@dataclass(frozen=True)
class PhaseLoading:
    training_batch_size: int
    validation_batch_size: int
    window_stride: int | None
    curriculum_column: str | None

    def batch_size(self, split: Literal["training", "validation"]) -> int:
        return (
            self.training_batch_size
            if split == "training"
            else self.validation_batch_size
        )


class CompiledSupervisedPhase(ResolvedTrainingPhase):
    """Run-owned phase. Builders replace the record when resolving runtime facts."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(exclude=True)
    seed: int = Field(exclude=True)

    @property
    def loading(self) -> PhaseLoading:
        from sequifier.evaluation.compatibility import evaluation_batch_size

        return PhaseLoading(
            self.batch_size,
            evaluation_batch_size(self),
            self.learning.window_stride,
            self.curriculum.column if self.curriculum else None,
        )

    def batch_snapshot_due(self, batches_since_snapshot: int) -> bool:
        interval = self.checkpoints.snapshot_interval_batches
        return interval is not None and batches_since_snapshot >= interval

    def ends_update(
        self, accumulated: int, last_batch: bool, batch_snapshot: bool
    ) -> bool:
        return accumulated >= self.accumulation_batches or last_batch or batch_snapshot


@dataclass(frozen=True)
class SupervisedRunPlan:
    phases: tuple[CompiledSupervisedPhase, ...]

    @classmethod
    def from_config(cls, config: ResolvedSequifierConfig) -> "SupervisedRunPlan":
        phases = []
        for index, spec in enumerate(config.training_plan):
            if not isinstance(spec, ResolvedTrainingPhase):
                raise ValueError(
                    "Native supervised training requires supervised phases"
                )
            phases.append(
                CompiledSupervisedPhase(
                    **dict(spec.model_copy(deep=True)),
                    index=index,
                    seed=spec.resolved_seed
                    if spec.resolved_seed is not None
                    else config.seed + index,
                )
            )
        return cls(tuple(phases))
