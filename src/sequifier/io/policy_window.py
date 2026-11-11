"""Policy alignment over the shared named-tensor and model-window contracts."""

from collections.abc import Mapping, Sequence

from sequifier.config.policy_config import ResolvedPolicyBinding
from sequifier.config.policy_inputs import PolicyEpisodeSpec
from sequifier.helpers import ModelWindowView, StoredWindowLayout, resolve_window_view
from sequifier.io.batch import SequifierInputs
from sequifier.io.preprocessing import normalize_values

PREVIOUS_ACTION = "__policy_previous_action"
DESIRED_RETURN = "__policy_return"


def normalize_policy_features(features, metadata, interface):
    """Apply persisted policy transforms identically in training and prediction."""
    features = dict(features)
    mask = metadata["attention_valid_mask"]
    if interface.policy.return_conditioned and interface.policy.input_encoding == "raw":
        features[DESIRED_RETURN] = (
            features[DESIRED_RETURN] / interface.policy.returns.scale
        )
    for column, statistics in interface.selected_columns_statistics.items():
        if column not in features or (not interface.normalize_real_columns):
            continue
        if interface.policy.input_encoding == "model":
            continue
        value = normalize_values(features[column], statistics)
        if value.ndim >= 2 and value.shape[:2] == mask.shape:
            valid = mask.reshape(*mask.shape, *([1] * (value.ndim - 2)))
            value = value.masked_fill(~valid, 0.0)
        features[column] = value
    return features


def action_index(value, config: PolicyEpisodeSpec) -> int | dict[str, int | float]:
    from sequifier.io.actions import encode_action

    return encode_action(value, config)


def encode_feature(value, column: str, config: PolicyEpisodeSpec):
    from sequifier.io.policy_materialization import encode_value

    return encode_value(value, column, config)


class PolicyWindowView:
    """At decision t, expose o_t and a_(t-1), and supervise a_t separately.

    Previous actions are decoder-local IDs at this boundary. They are converted
    to the persisted input vocabulary here; the start token is input-only.
    """

    def __init__(self, config: ResolvedPolicyBinding):
        self.config = config
        self.view = ModelWindowView(config.context_length, "policy", 0)
        self.resolved = resolve_window_view(
            StoredWindowLayout(config.context_length, 0, 2), self.view
        )

    def episode_inputs(
        self, episode, step: int, returns, action_ids=None
    ) -> SequifierInputs:
        """Expose one causal decision from a validated episode in either lifecycle."""
        if self.config.return_conditioned and returns is None:
            from sequifier.io.episode_features import return_labels

            length = episode.return_length(self.config.returns.finite_horizon)
            if length is None:
                raise ValueError("conditioned likelihood requires a known reward tail")
            returns = return_labels(episode, length, self.config.returns.gamma)
        start = max(0, step + 1 - self.config.context_length)
        initial = (
            (None)
            if episode.starts_episode
            else action_index(episode.previous_action, self.config)
        )
        previous = [initial] if start == 0 else []
        if action_ids is None:
            previous += [
                action_index(value, self.config)
                for value in episode.actions[max(0, start - 1) : step]
            ]
        else:
            previous += action_ids[max(0, start - 1) : step]
        return self.inputs(
            [episode.observation(position) for position in range(start, step + 1)],
            previous,
            returns[start : step + 1] if returns is not None else None,
            episode.static_inputs,
        )

    def successor_inputs(self, episode, step: int, successor) -> SequifierInputs:
        """Build h_(t+1) from the true successor and action a_t."""
        start = max(0, step + 2 - self.config.context_length)
        observations = [
            episode.observation(position) for position in range(start, step + 1)
        ] + [successor]
        previous = [
            (
                (None)
                if episode.starts_episode
                else action_index(episode.previous_action, self.config)
            )
            if position == 0
            else action_index(episode.actions[position - 1], self.config)
            for position in range(start, step + 2)
        ]
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
            budgets = returns[start : step + 1] + [following]
        return self.inputs(observations, previous, budgets, episode.static_inputs)

    def inputs(
        self,
        observations: Sequence[Mapping],
        previous_actions: Sequence[int | dict[str, int | float] | None],
        desired_returns: Sequence[float] | None,
        static_inputs: Mapping,
    ) -> SequifierInputs:
        config = self.config
        from sequifier.io.policy_materialization import materialize_history

        if config.return_conditioned:
            if desired_returns is None or len(desired_returns) != len(observations):
                raise ValueError(
                    "return-conditioned histories require one return per observation"
                )
            observations = [
                {**row, DESIRED_RETURN: value}
                for row, value in zip(observations, desired_returns)
            ]
        return materialize_history(
            config, observations, previous_actions, static_inputs
        )
