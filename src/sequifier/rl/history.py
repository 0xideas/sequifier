"""One causal policy window builder shared by offline training and execution."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from sequifier.config.policy_config import CategoricalPolicyConfig
from sequifier.io.batch import SequifierInputs, collate_model_inputs
from sequifier.io.policy_window import PolicyWindowView, action_index  # noqa: F401


@dataclass(frozen=True)
class PolicyInputs:
    observations: Tensor
    previous_actions: Tensor
    attention_valid_mask: Tensor
    static_inputs: Tensor
    desired_returns: Tensor | None

    def to(self, device: str | torch.device) -> "PolicyInputs":
        return PolicyInputs(
            self.observations.to(device),
            self.previous_actions.to(device),
            self.attention_valid_mask.to(device),
            self.static_inputs.to(device),
            self.desired_returns.to(device)
            if self.desired_returns is not None
            else None,
        )


def collate_policy_inputs(inputs: Sequence[PolicyInputs | SequifierInputs]):
    if not inputs:
        raise ValueError("cannot collate empty policy inputs")
    named = [value for value in inputs if isinstance(value, SequifierInputs)]
    if named:
        if len(named) != len(inputs):
            raise ValueError("cannot mix named and legacy dense policy inputs")
        return collate_model_inputs(named)
    legacy = [value for value in inputs if isinstance(value, PolicyInputs)]
    conditioned = legacy[0].desired_returns is not None
    if any((value.desired_returns is not None) != conditioned for value in legacy):
        raise ValueError("policy batches cannot mix conditioning modes")
    return PolicyInputs(
        observations=torch.cat([value.observations for value in legacy]),
        previous_actions=torch.cat([value.previous_actions for value in legacy]),
        attention_valid_mask=torch.cat(
            [value.attention_valid_mask for value in legacy]
        ),
        static_inputs=torch.cat([value.static_inputs for value in legacy]),
        desired_returns=(
            torch.cat(
                [
                    value.desired_returns
                    for value in legacy
                    if value.desired_returns is not None
                ]
            )
            if conditioned
            else None
        ),
    )


def build_policy_inputs(
    config: CategoricalPolicyConfig,
    observations: Sequence[Mapping],
    previous_actions: Sequence[int],
    desired_returns: Sequence[float] | None,
    static_inputs: Mapping,
) -> SequifierInputs:
    """Compatibility entry point for the shared policy window view."""
    return PolicyWindowView(config).inputs(
        observations, previous_actions, desired_returns, static_inputs
    )
