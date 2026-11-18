"""Read-only phase views for native supervised execution."""

from dataclasses import dataclass
from typing import Literal

from sequifier.config.train_config import (
    ResolvedDatasetTrainingSpec,
    ResolvedSequifierConfig,
    ResolvedTrainingPhase,
    TrainingRuntimeSettings,
)


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


@dataclass(frozen=True)
class SupervisedPhaseView:
    index: int
    seed: int
    loading: PhaseLoading
    _spec: ResolvedTrainingPhase

    @classmethod
    def from_spec(
        cls, index: int, seed: int, spec: ResolvedTrainingPhase
    ) -> "SupervisedPhaseView":
        from sequifier.evaluation.compatibility import evaluation_batch_size

        return cls(
            index=index,
            seed=seed,
            loading=PhaseLoading(
                spec.batch_size,
                evaluation_batch_size(spec),
                spec.learning.window_stride,
                spec.curriculum.column if spec.curriculum else None,
            ),
            _spec=spec.model_copy(deep=True),
        )

    def spec(self) -> ResolvedTrainingPhase:
        """Give consumers their own copy of the configuration models."""
        return self._spec.model_copy(deep=True)

    def with_spec(self, spec: ResolvedTrainingPhase) -> "SupervisedPhaseView":
        return self.from_spec(self.index, self.seed, spec)

    def objective_settings(
        self, settings: TrainingRuntimeSettings
    ) -> TrainingRuntimeSettings:
        learning = self._spec.learning
        return settings.model_copy(
            update={
                "bert_spec": learning.bert_spec,
                "next_occurrence_config": learning.next_occurrence_config,
                "window_stride": self.loading.window_stride,
            }
        ).model_copy(deep=True)


@dataclass(frozen=True)
class SupervisedRunPlan:
    phases: tuple[SupervisedPhaseView, ...]

    @classmethod
    def from_config(cls, config: ResolvedSequifierConfig) -> "SupervisedRunPlan":
        phases = []
        for index, spec in enumerate(config.training_plan):
            if not isinstance(spec, ResolvedTrainingPhase):
                raise ValueError(
                    "Native supervised training requires supervised phases"
                )
            phases.append(
                SupervisedPhaseView.from_spec(index, config.seed + index, spec)
            )
        return cls(tuple(phases))


def dataset_learning_view(
    dataset: ResolvedDatasetTrainingSpec, phase: ResolvedTrainingPhase
) -> ResolvedDatasetTrainingSpec:
    learning = phase.learning.datasets.get(dataset.name)
    return dataset.model_copy(update=learning.model_dump()) if learning else dataset
