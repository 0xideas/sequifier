"""Elementwise clipped actor objective, independent of algorithm weighting."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ClippedPolicyObjective:
    loss: Tensor
    ratio: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor


def clipped_policy_objective(log_probability, old_log_probability, advantages, epsilon):
    log_ratio = log_probability.float() - old_log_probability.float()
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon)
    return ClippedPolicyObjective(
        loss=-torch.minimum(ratio * advantages, clipped * advantages),
        ratio=ratio,
        approximate_kl=ratio - 1 - log_ratio,
        clip_fraction=((ratio - 1).abs() > epsilon).float(),
    )
