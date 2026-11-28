"""Policy alignment over the shared named-tensor and model-window contracts."""

from collections.abc import Mapping, Sequence

import torch

from sequifier.config.column_names import NEXT_ACTION_OFFSET
from sequifier.config.policy_config import ResolvedPolicyBinding
from sequifier.config.policy_inputs import PolicyEpisodeSpec
from sequifier.contract_versions import ContractVersions
from sequifier.helpers import ModelWindowView, StoredWindowLayout, resolve_window_view
from sequifier.io.batch import ModelEncodedPolicyInputs, SequifierInputs
from sequifier.io.policy_materialization import DESIRED_RETURN, materialize_history
from sequifier.io.preprocessing import normalize_values

PREVIOUS_ACTION = "__policy_previous_action"


def normalize_policy_features(inputs, interface, *, input_encoding=None):
    """Apply transforms to policy histories that still contain raw real values."""
    features = dict(inputs.inputs)
    if input_encoding == "model" or isinstance(inputs, ModelEncodedPolicyInputs):
        return features
    mask = inputs.attention_valid_mask
    if interface.policy.return_conditioned:
        features[DESIRED_RETURN] = (
            features[DESIRED_RETURN] / interface.policy.returns.scale
        )
    for column, statistics in interface.selected_columns_statistics.items():
        if column not in features or not interface.normalize_real_columns:
            continue
        value = normalize_values(features[column], statistics)
        if value.ndim >= 2 and value.shape[:2] == mask.shape:
            valid = mask.reshape(*mask.shape, *([1] * (value.ndim - 2)))
            value = value.masked_fill(~valid, 0.0)
        features[column] = value
    return features


def checked_policy_features(inputs, interface, *, input_encoding=None):
    features = normalize_policy_features(
        inputs, interface, input_encoding=input_encoding
    )
    if any(
        tensor.is_floating_point() and not torch.isfinite(tensor).all()
        for tensor in features.values()
    ):
        raise ValueError("frozen input transforms produced non-finite values")
    return features


def action_index(value, config: PolicyEpisodeSpec) -> int | dict[str, int | float]:
    from sequifier.io.actions import encode_action

    return encode_action(value, config)


def encode_feature(value, column: str, config: PolicyEpisodeSpec):
    from sequifier.io.policy_materialization import encode_value

    return encode_value(value, column, config)


class PolicyWindowView:
    """History ends at row i. The executed action label comes from row i + 1.

    Previous actions are decoder-local IDs at this boundary. They are converted
    to the persisted input vocabulary here; the start token is input-only.
    """

    def __init__(self, config: ResolvedPolicyBinding):
        self.config = config
        self._categorical_codecs = {}
        self.view = ModelWindowView(config.context_length, "policy", NEXT_ACTION_OFFSET)
        self.resolved = resolve_window_view(
            StoredWindowLayout(
                config.context_length + NEXT_ACTION_OFFSET,
                NEXT_ACTION_OFFSET,
                ContractVersions.STORED_WINDOW_LAYOUT,
            ),
            self.view,
        )

    def episode_inputs(self, episode, step: int, returns) -> SequifierInputs:
        """Expose one causal decision from a validated episode in either lifecycle."""
        if self.config.return_conditioned and returns is None:
            from sequifier.io.episode_features import return_labels

            length = episode.return_length(self.config.returns.finite_horizon)
            if length is None:
                raise ValueError("conditioned likelihood requires a known reward tail")
            returns = return_labels(episode, length, self.config.returns.gamma)
        start = max(0, step + 1 - self.config.context_length)
        previous = [
            action_index(value, self.config) if value is not None else None
            for value in episode.incoming_actions[start : step + 1]
        ]
        return self._with_context(
            episode,
            [episode.observation(position) for position in range(start, step + 1)],
            previous,
            returns[start : step + 1] if returns is not None else None,
            list(episode.observation_masks[start : step + 1]),
        )

    def successor_inputs(self, episode, step: int) -> SequifierInputs:
        """Read the next observation with its incoming action."""
        budgets = None
        if self.config.return_conditioned:
            from sequifier.io.episode_features import return_labels
            from sequifier.trajectory.returns import remaining_return

            length = episode.return_length(self.config.returns.finite_horizon)
            if length is None:
                raise ValueError("conditioned successors require a known reward tail")
            returns = return_labels(episode, length, self.config.returns.gamma)
            following = remaining_return(
                returns[step],
                episode.transitions.rewards[step],
                self.config.returns.gamma,
            )
            budgets = [*returns[: step + 1], following]
        return self.episode_inputs(episode, step + 1, budgets)

    def _with_context(self, episode, observations, previous, budgets, masks):
        context = episode.context
        needed = self.config.context_length - len(observations)
        if context is not None and needed > 0:
            prefix = context[-needed:]
            observations = [step.observation for step in prefix] + observations
            masks = [step.masked for step in prefix] + masks
            lagged = [step.previous_action for step in prefix]
            previous = [
                action_index(a, self.config) if a is not None else None for a in lagged
            ] + previous
            if budgets is not None:
                earlier = []
                remaining = budgets[0]
                for step in reversed(prefix):
                    remaining = step.return_before(remaining, self.config.returns.gamma)
                    earlier.append(remaining)
                budgets = earlier[::-1] + list(budgets)
        return self.inputs(
            observations, previous, budgets, episode.static_inputs, masks=masks
        )

    def inputs(
        self,
        observations: Sequence[Mapping],
        previous_actions: Sequence[int | dict[str, int | float] | None],
        desired_returns: Sequence[float] | None,
        static_inputs: Mapping,
        *,
        masks=None,
    ) -> SequifierInputs:
        config = self.config
        if config.return_conditioned:
            if desired_returns is None or len(desired_returns) != len(observations):
                raise ValueError(
                    "return-conditioned histories require one return per observation"
                )
            start = max(0, len(observations) - config.context_length)
            observations = [
                {**row, DESIRED_RETURN: value}
                for row, value in zip(observations[start:], desired_returns[start:])
            ]
        return materialize_history(
            config,
            observations,
            previous_actions,
            static_inputs,
            codecs=self._categorical_codecs,
            masks=masks,
        )
