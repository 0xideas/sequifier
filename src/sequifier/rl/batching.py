"""Batch mechanics without algorithm-specific weighting or sampling policies."""

from collections.abc import Iterable, Iterator
from typing import Any, TypeVar, cast

import torch

from sequifier.io.tensors import TensorRecord


def batches(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def optimize_groups(session, count, group_kl, backward):
    """Check KL before accumulating each shuffled group and stepping its optimizers."""
    phase = session.context.phase
    size = phase.batch_size * phase.accumulation_batches
    norm_sums = {}
    steps = 0
    stopped = False
    epochs = 0
    for epochs in range(1, phase.optimization_epochs + 1):
        order = torch.randperm(count, generator=session.sampler).tolist()
        for group in batches(order, size):
            if phase.old_policy_target_kl is not None:
                with torch.no_grad():
                    stopped = group_kl(group) > phase.old_policy_target_kl
                if stopped:
                    break
            session.zero_grad()
            backward(group, epochs)
            for role, norm in session.step().items():
                norm_sums[role] = norm_sums.get(role, 0.0) + norm
            steps += 1
        if stopped:
            break
    return epochs, stopped, {role: total / steps for role, total in norm_sums.items()}


class MetricTotals(dict):
    def __init__(self, *names):
        super().__init__(dict.fromkeys(names, 0.0))

    def add(self, values, divisor=1):
        for name in self:
            self[name] += float(values[name].detach().sum()) / divisor

    def means(self, count):
        return {name: total / count for name, total in self.items()}


RolloutBatch = TypeVar("RolloutBatch", bound=tuple[Any, ...])


class RolloutPartition:
    """Map a common shuffled rollout order to the rank that owns each item."""

    def __init__(self, strategy, size):
        self.strategy = strategy
        counts = strategy.gather_objects(size)
        self.start = sum(counts[: strategy.rank])
        self.end = self.start + size
        self.total = sum(counts)

    def local(self, group: Iterable[int]) -> list[int]:
        return [i - self.start for i in group if self.start <= i < self.end]

    def aligned(
        self, chunks: Iterable[RolloutBatch]
    ) -> Iterator[tuple[RolloutBatch, bool]]:
        """Exchange one valid batch for zero-weight backwards on idle ranks."""
        chunks = iter(chunks)
        while True:
            batch = next(chunks, None)
            active = self.strategy.gather_objects(batch is not None)
            if not any(active):
                return
            if not all(active):
                owner = active.index(True)
                template = None
                if self.strategy.rank == owner:
                    assert batch is not None
                    template = _move_rollout_batch(batch, "cpu")
                    if self.strategy.device.type == "cuda":
                        torch.cuda.current_stream(self.strategy.device).synchronize()
                template = self.strategy.broadcast(template, owner)
                if batch is None:
                    yield _move_rollout_batch(template, self.strategy.device), False
                    continue
            assert batch is not None
            yield batch, True


def _move_rollout_batch(batch: RolloutBatch, device) -> RolloutBatch:
    moved = []
    for item in batch:
        if isinstance(item, dict):
            item = {name: tensor.to(device) for name, tensor in item.items()}
        elif isinstance(item, (TensorRecord, torch.Tensor)):
            item = item.to(device)
        moved.append(item)
    return cast(RolloutBatch, tuple(moved))
