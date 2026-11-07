"""One causal policy window builder shared by offline training and execution."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from sequifier.config.policy_config import CategoricalPolicyConfig
from sequifier.rollout.history import numeric_fields, numeric_scalar


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


def collate_policy_inputs(inputs: Sequence[PolicyInputs]) -> PolicyInputs:
    if not inputs:
        raise ValueError("cannot collate empty policy inputs")
    conditioned = inputs[0].desired_returns is not None
    if any((value.desired_returns is not None) != conditioned for value in inputs):
        raise ValueError("policy batches cannot mix conditioning modes")
    return PolicyInputs(
        observations=torch.cat([value.observations for value in inputs]),
        previous_actions=torch.cat([value.previous_actions for value in inputs]),
        attention_valid_mask=torch.cat(
            [value.attention_valid_mask for value in inputs]
        ),
        static_inputs=torch.cat([value.static_inputs for value in inputs]),
        desired_returns=(
            torch.cat(
                [
                    value.desired_returns
                    for value in inputs
                    if value.desired_returns is not None
                ]
            )
            if conditioned
            else None
        ),
    )


def action_index(value: float, config: CategoricalPolicyConfig) -> int:
    value = numeric_scalar(value, "recorded action")
    try:
        return config.action_values.index(value)
    except ValueError as exc:
        raise ValueError(
            f"action {value!r} is outside action_values {config.action_values!r}"
        ) from exc


def build_policy_inputs(
    config: CategoricalPolicyConfig,
    observations: Sequence[Mapping[str, float]],
    previous_actions: Sequence[int],
    desired_returns: Sequence[float] | None,
    static_inputs: Mapping[str, float],
) -> PolicyInputs:
    """Only supplied decision prefixes are visible; targets are never an argument.

    At position t, observations[t] is o_t and previous_actions[t] is a_(t-1).
    The action-count ID is the explicit start token, never an executable action.
    """
    length = len(observations)
    if not length or len(previous_actions) != length:
        raise ValueError(
            "policy observations and previous actions must have matching nonempty lengths"
        )
    if config.return_conditioned and (
        desired_returns is None or len(desired_returns) != length
    ):
        raise ValueError(
            "return-conditioned windows require one desired return per decision position"
        )
    static = numeric_fields(
        static_inputs, set(config.static_columns), "policy static inputs"
    )
    context = config.context_length
    start = max(0, length - context)
    count = length - start
    padding = context - count
    feature_rows = [[0.0] * len(config.observation_columns) for _ in range(padding)]
    for observation in observations[start:]:
        values = numeric_fields(
            observation, set(config.observation_columns), "policy observation"
        )
        for column, count in config.observation_n_classes.items():
            value = values[column]
            if value != int(value) or not 0 <= value < count:
                raise ValueError(
                    f"{column}: observation must be a categorical ID in [0, {count})"
                )
        feature_rows.append([values[column] for column in config.observation_columns])
    action_ids = [len(config.action_values)] * padding
    for action in previous_actions[start:]:
        if (
            isinstance(action, bool)
            or not isinstance(action, int)
            or not 0 <= action <= len(config.action_values)
        ):
            raise ValueError(
                "previous actions must be categorical IDs or the start token"
            )
        action_ids.append(action)
    returns = None
    if config.return_conditioned:
        assert desired_returns is not None
        returns = torch.tensor(
            [
                [0.0] * padding
                + [
                    numeric_scalar(value, "desired_return")
                    for value in desired_returns[start:]
                ]
            ],
            dtype=torch.float32,
        )
    inputs = PolicyInputs(
        observations=torch.tensor([feature_rows], dtype=torch.float32),
        previous_actions=torch.tensor([action_ids], dtype=torch.int64),
        attention_valid_mask=torch.tensor([[False] * padding + [True] * count]),
        static_inputs=torch.tensor(
            [[static[column] for column in config.static_columns]], dtype=torch.float32
        ),
        desired_returns=returns,
    )
    if (
        not torch.isfinite(inputs.observations).all()
        or not torch.isfinite(inputs.static_inputs).all()
        or (returns is not None and not torch.isfinite(returns).all())
    ):
        raise ValueError("policy inputs exceed the finite float32 range")
    return inputs
