"""Structured metric output owned by the run runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sequifier.evaluation.service import EvaluationResult
from sequifier.training.metrics import StructuredMetricWriters


@dataclass(frozen=True)
class TrainingMetrics:
    dataset: str
    part: str | None
    epoch: int
    batch: int
    batches_total: int
    global_step: int
    window_batches: int
    total_loss: float
    target_losses: dict[str, float]
    learning_rate: float
    seconds_per_batch: float


class MetricsService:
    def __init__(self, context: Any, config: Any) -> None:
        self.context = context
        dataset_count = len(config.dataset_training)
        evaluated = {source.dataset for source in config.evaluation_sources}
        self.writers = (
            {
                name: StructuredMetricWriters(
                    config.project_root,
                    config.model_name,
                    context.rank,
                    class_share_columns=dataset.class_share_log_columns,
                    dataset_name=name,
                    dataset_count=dataset_count,
                    validation_enabled=name in evaluated,
                )
                for name, dataset in config.dataset_training.items()
            }
            if context.rank == 0
            else {}
        )

    def record_training(self, result: TrainingMetrics) -> None:
        if self.context.rank != 0:
            return
        self.writers[result.dataset].write_training(
            run_id=self.context.run_id,
            session_id=self.context.session_id,
            epoch=result.epoch,
            batch=result.batch,
            batches_total=result.batches_total,
            global_step=result.global_step,
            window_batches=result.window_batches,
            total_loss=result.total_loss,
            target_losses=result.target_losses,
            learning_rate=result.learning_rate,
            seconds_per_batch=result.seconds_per_batch,
            dataset=result.dataset,
            part=result.part,
        )

    def record_evaluation(
        self, result: EvaluationResult, *, context: Any, learning_rate: float
    ) -> None:
        if self.context.rank != 0:
            return
        for source in result.sources.values():
            self.writers[source.dataset].write_validation(
                run_id=context.run_id,
                session_id=context.session_id,
                evaluation_kind=context.kind,
                epoch=context.epoch,
                batch=context.training_batch,
                batches_total=context.training_batches_total,
                global_step=context.global_step,
                total_loss=source.total_loss,
                target_losses=source.target_losses,
                baseline_loss=source.baseline_loss,
                baseline_target_losses=source.baseline_target_losses,
                class_distributions=source.class_distributions,
                learning_rate=learning_rate,
                elapsed_seconds=source.elapsed_seconds,
                dataset=source.dataset,
                part=source.part,
            )
