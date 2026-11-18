"""Preflight scheduler budgets using the same source ordering as training."""

from dataclasses import dataclass

import torch

from sequifier.config.train_config import (
    ResolvedDatasetTrainingSpec,
    ResolvedTrainingSource,
)
from sequifier.io.batch import SequifierBatch
from sequifier.training.plan import SupervisedRunPlan
from sequifier.training.runtime import RuntimeBatch, SourceScheduler, source_part_names


@dataclass
class _CountedSource:
    """Supply batch identities to the scheduler without consuming training data."""

    config: ResolvedTrainingSource
    dataset_config: ResolvedDatasetTrainingSpec
    batches: int

    def set_epoch(self, epoch, split):
        pass

    def iter_batches(self, split="training"):
        batch = RuntimeBatch(
            self.dataset_config.name,
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


def _phase_update_budget(config, parts, view):
    phase = view.spec()
    counts = {}
    sources = []
    for source in phase.sources:
        dataset = config.dataset_training[source.dataset]
        batches = 0
        for part in source_part_names(source, dataset):
            key = (source.dataset, part)
            if key not in counts:
                counts[key] = parts[source.dataset][
                    part
                ].factory.count_training_batches(view)
            batches += counts[key]
        sources.append(_CountedSource(source, dataset, batches))
    scheduler = SourceScheduler(view, sources)
    updates = 0
    for epoch in range(1, phase.epochs + 1):
        active_dataset, accumulated = None, 0
        for batch in scheduler.iter_epoch(epoch):
            if active_dataset != batch.dataset:
                updates += int(accumulated > 0)
                accumulated = 0
            active_dataset = batch.dataset
            accumulated += 1
            if accumulated == phase.accumulation_batches:
                updates += 1
                accumulated = 0
        updates += int(accumulated > 0)
    return updates


def resolve_scheduler_budgets(plan, config, parts):
    """Resolve applied-update budgets, including dataset-boundary flushes."""
    views = list(plan.phases)
    phases = [view.spec() for view in views]
    phase_budgets = {}
    for index, phase in enumerate(phases):
        role = phase.optimization["model"]
        if (
            role.scheduler.name != "OneCycleLR"
            or role.scheduler_step_on != "optimizer_step"
            or role.scheduler.arguments.get("total_steps") is not None
        ):
            continue
        start = index
        while start > 0 and phases[start].optimizer_state == "continue":
            start -= 1
        end = index + 1
        while end < len(phases) and phases[end].optimizer_state == "continue":
            end += 1
        for selected in range(start, end):
            if selected not in phase_budgets:
                phase_budgets[selected] = _phase_update_budget(
                    config, parts, views[selected]
                )
        budget = sum(phase_budgets[selected] for selected in range(start, end))
        if budget <= 0:
            raise ValueError(
                f"{phase.name}: OneCycleLR requires a positive optimizer-update budget"
            )
        scheduler = role.scheduler.model_copy(
            update={"arguments": {**role.scheduler.arguments, "total_steps": budget}}
        )
        optimization = {
            **phase.optimization,
            "model": role.model_copy(update={"scheduler": scheduler}),
        }
        views[index] = views[index].with_spec(
            phase.model_copy(update={"optimization": optimization})
        )
    return SupervisedRunPlan(tuple(views))
