"""Batch mechanics without algorithm-specific weighting or sampling policies."""

import torch


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
            backward(group)
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
