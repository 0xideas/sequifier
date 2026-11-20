"""Preflight scheduler budgets using the same source ordering as training."""

from itertools import repeat

from sequifier.runtime.plan import continuation_groups
from sequifier.training.plan import SupervisedRunPlan
from sequifier.training.runtime import SourceScheduler


def _phase_update_budget(config, parts, phase):
    counts = {}
    sources = []
    for source in phase.sources:
        batches = 0
        for part in source.parts:
            key = (source.dataset, part)
            if key not in counts:
                counts[key] = parts[source.dataset][
                    part
                ].factory.count_training_batches(phase)
            batches += counts[key]
        sources.append((source.dataset, batches))
    scheduler = SourceScheduler(phase, config.dataset_training)
    total_batches = sum(count for _, count in sources)
    since_snapshot = 0
    updates = 0
    for epoch in range(1, phase.epochs + 1):
        active_dataset, accumulated = None, 0
        streams = [repeat(dataset, count) for dataset, count in sources]
        for batch, dataset in enumerate(scheduler.iter_epoch(epoch, streams), 1):
            if active_dataset != dataset:
                updates += int(accumulated > 0)
                accumulated = 0
            active_dataset = dataset
            accumulated += 1
            since_snapshot += 1
            snapshot = phase.batch_snapshot_due(since_snapshot)
            if snapshot:
                since_snapshot = 0
            if phase.ends_update(accumulated, batch == total_batches, snapshot):
                updates += 1
                accumulated = 0
    return updates


def resolve_scheduler_budgets(plan, config, parts):
    """Resolve applied-update budgets, including dataset-boundary flushes."""
    views = list(plan.phases)
    for start, group in continuation_groups(views):
        budget = None
        for index, phase in enumerate(group, start):
            role = phase.optimization["model"]
            if (
                role.scheduler.name != "OneCycleLR"
                or role.scheduler_step_on != "optimizer_step"
                or role.scheduler.arguments.get("total_steps") is not None
            ):
                continue
            if budget is None:
                budget = sum(
                    _phase_update_budget(config, parts, item) for item in group
                )
            optimization = {**phase.optimization, "model": role.resolve_budget(budget)}
            views[index] = phase.model_copy(update={"optimization": optimization})
    return SupervisedRunPlan(tuple(views))
