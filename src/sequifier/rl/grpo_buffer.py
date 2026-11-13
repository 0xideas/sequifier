"""Bounded complete groups with reconstructable causal frames and fixed scores."""

from dataclasses import dataclass
from typing import cast

import torch

from sequifier.io.actions import action_batch
from sequifier.io.batch import SequifierInputs, collate_model_inputs
from sequifier.rl.history import history_inputs


@dataclass(frozen=True)
class GRPOTrajectory:
    group_index: int
    sample_index: int
    start_token: object
    episode_id: str
    frames: tuple
    context_length: int
    actions: tuple[int | dict, ...]
    rewards: tuple[float, ...]
    old_log_probabilities: tuple[float, ...]
    reference_log_probabilities: tuple[float, ...]
    cutoff_reason: str
    score: float
    advantage: float
    generation: int

    def inputs(self, spec, step):
        return history_inputs(
            spec, self.frames[max(0, step + 1 - spec.context_length) : step + 1]
        )

    def batches(self, actor, chunk_size):
        for start in range(0, len(self.actions), chunk_size):
            end = min(start + chunk_size, len(self.actions))
            yield (
                cast(
                    SequifierInputs,
                    collate_model_inputs(
                        [self.inputs(actor.config, t) for t in range(start, end)]
                    ),
                ).to(actor.device),
                action_batch(
                    self.actions[start:end], actor.config, device=actor.device
                ),
                torch.tensor(
                    self.old_log_probabilities[start:end],
                    dtype=torch.float32,
                    device=actor.device,
                ),
                torch.tensor(
                    self.reference_log_probabilities[start:end],
                    dtype=torch.float32,
                    device=actor.device,
                ),
            )

    def as_dict(self):
        return {
            **self.__dict__,
            "frames": [frame.state_dict() for frame in self.frames],
            "decision_mask": [True] * len(self.actions),
            "history_lengths": [
                min(i + 1, self.context_length) for i in range(len(self.actions))
            ],
        }


class SealedGroupBuffer:
    def __init__(
        self, groups, generation, group_size, reference_identity, estimated_bytes
    ):
        if not groups or any(len(group) != group_size for group in groups):
            raise ValueError("GRPO requires complete groups")
        self.groups, self.generation = tuple(tuple(g) for g in groups), generation
        self.reference_identity, self.estimated_bytes = (
            reference_identity,
            estimated_bytes,
        )
        self.trajectories = tuple(
            trajectory for group in self.groups for trajectory in group
        )
        self.consumed = False
        if any(t.generation != generation or not t.actions for t in self.trajectories):
            raise ValueError("GRPO group buffer has stale/empty trajectories")

        indices = [group[0].group_index for group in self.groups]
        if len(set(indices)) != len(indices):
            raise ValueError("GRPO buffer repeats a group identity")
        for group in self.groups:
            if {t.sample_index for t in group} != set(range(group_size)) or any(
                t.group_index != group[0].group_index
                or t.start_token != group[0].start_token
                for t in group
            ):
                raise ValueError("GRPO group has inconsistent start/sample identities")

    def require_fresh(self, generation, reference_identity):
        if (
            self.consumed
            or generation != self.generation
            or reference_identity != self.reference_identity
        ):
            raise ValueError(
                "GRPO buffer is consumed, stale or has a different reference"
            )

    def consume(self):
        if self.consumed:
            raise ValueError("GRPO buffer has already been consumed")
        self.consumed = True
