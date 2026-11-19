"""Resumable dataset permutations at complete accumulation boundaries."""

import torch


class BatchOrder:
    def __init__(self, size, group_size, seed, *, sample=None, state=None):
        if size < 1 or group_size < 1:
            raise ValueError("batch order requires positive dataset and group sizes")
        self.size, self.group_size, self.sample = size, group_size, sample
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch, self.position, self.order = 0, 0, None
        if state is not None:
            self.epoch = state["epoch"]
            self.position = state["position"]
            self.order = state["order"]
            self._validate()
            self.generator.set_state(state["generator"])

    @property
    def transactions(self):
        per_epoch = (self.size + self.group_size - 1) // self.group_size
        return self.epoch * per_epoch + self.position // self.group_size

    def _validate(self):
        if (
            not isinstance(self.epoch, int)
            or isinstance(self.epoch, bool)
            or self.epoch < 0
        ):
            raise ValueError("checkpoint sampler epoch is invalid")
        if (
            not isinstance(self.position, int)
            or isinstance(self.position, bool)
            or not 0 <= self.position < self.size
            or self.position % self.group_size
        ):
            raise ValueError(
                "checkpoint sampler is not a complete accumulation boundary"
            )
        if self.order is None:
            if self.position:
                raise ValueError("checkpoint sampler has a position without an order")
        elif (
            not isinstance(self.order, list)
            or any(
                not isinstance(index, int) or isinstance(index, bool)
                for index in self.order
            )
            or sorted(self.order) != list(range(self.size))
        ):
            raise ValueError("checkpoint sampler order is not the dataset permutation")
        elif not self.position:
            raise ValueError("checkpoint sampler has an unconsumed order")

    def next(self):
        if self.order is None:
            self.order = (
                list(self.sample(self.generator))
                if self.sample is not None
                else torch.randperm(self.size, generator=self.generator).tolist()
            )
        group = self.order[self.position : self.position + self.group_size]
        self.position += len(group)
        if self.position == self.size:
            self.epoch += 1
            self.position, self.order = 0, None
        return group

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "position": self.position,
            "order": self.order,
            "generator": self.generator.get_state(),
        }
