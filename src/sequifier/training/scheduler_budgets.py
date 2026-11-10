"""Preflight scheduler budgets using the same source ordering as training."""

from dataclasses import dataclass

import torch

from sequifier.config.train_config import ResolvedTrainingSource
from sequifier.io.batch import SequifierBatch
from sequifier.training.runtime import (
    DatasetRuntime,
    RuntimeBatch,
    SourceScheduler,
    build_source_runtime,
)


@dataclass
class _CountedSource:
    """Supply batch identities to the scheduler without consuming training data."""

    config: ResolvedTrainingSource
    dataset: DatasetRuntime
    batches: int

    def set_epoch(self, epoch, split):
        pass

    def iter_batches(self, split="training"):
        batch = RuntimeBatch(
            self.dataset.name,
            self.config.part or "",
            SequifierBatch(
                {},
                {},
                {
                    "attention_valid_mask": torch.empty(0, dtype=torch.bool),
                    "target_valid_mask": torch.empty(0, dtype=torch.bool),
                },
            ),
        )
        for _ in range(self.batches):
            yield batch


def _phase_update_budget(config, datasets, phase_index):
    phase = config.training_plan[phase_index]
    for dataset in datasets.values():
        for part in dataset.parts.values():
            if part.factory.batch_size != phase.batch_size:
                part.factory.loaders.clear()
                part.factory.batch_size = phase.batch_size
    sources = [build_source_runtime(source, datasets) for source in phase.sources]
    counted = [
        _CountedSource(source.config, source.dataset, source.num_batches())
        for source in sources
    ]
    scheduler = SourceScheduler(
        phase, counted, seed=config.seed + phase_index, phase_index=phase_index
    )
    updates = 0
    for epoch in range(1, phase.epochs + 1):
        active_dataset, accumulated = None, 0
        for batch in scheduler.iter_epoch(epoch):
            if active_dataset != batch.dataset:
                updates += int(accumulated > 0)
                accumulated = 0
            active_dataset = batch.dataset
            accumulated += 1
            if accumulated == phase.accumulation_steps:
                updates += 1
                accumulated = 0
        updates += int(accumulated > 0)
    return updates


def validate_supervised_scheduler_budgets(config, datasets, active_phase_index):
    """Validate maximum applied-update budgets, including dataset-boundary flushes."""
    phase_budgets = {}
    for index, phase in enumerate(config.training_plan):
        role = phase.optimization["model"]
        if role.scheduler.name != "OneCycleLR" or role.scheduler_step_on != "batch":
            continue
        start = index
        while start > 0 and config.training_plan[start].optimizer_state == "continue":
            start -= 1
        end = index + 1
        while (
            end < len(config.training_plan)
            and config.training_plan[end].optimizer_state == "continue"
        ):
            end += 1
        for selected in range(start, end):
            if selected not in phase_budgets:
                phase_budgets[selected] = _phase_update_budget(
                    config, datasets, selected
                )
        budget = sum(phase_budgets[selected] for selected in range(start, end))
        if role.scheduler.arguments["total_steps"] != budget:
            raise ValueError(
                f"{phase.name}: OneCycleLR.total_steps must equal optimizer-update budget {budget}"
            )
    for dataset in datasets.values():
        for part in dataset.parts.values():
            if (
                part.factory.batch_size
                != config.training_plan[active_phase_index].batch_size
            ):
                part.factory.batch_size = config.training_plan[
                    active_phase_index
                ].batch_size
                part.factory.loaders.clear()
