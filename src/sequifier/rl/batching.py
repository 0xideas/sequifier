"""Batch mechanics without algorithm-specific weighting or sampling policies."""

import torch


def batches(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def shuffled_groups(size, group_size, generator):
    return batches(torch.randperm(size, generator=generator).tolist(), group_size)


class MetricTotals(dict):
    def __init__(self, *names):
        super().__init__(dict.fromkeys(names, 0.0))

    def add(self, values, divisor=1):
        for name in self:
            self[name] += float(values[name].detach().sum()) / divisor

    def means(self, count):
        return {name: total / count for name, total in self.items()}
