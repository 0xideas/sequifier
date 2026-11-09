"""Policy alignment over the shared named-tensor and model-window contracts."""

from collections.abc import Mapping, Sequence

import torch

from sequifier.config.policy_config import CategoricalPolicyConfig
from sequifier.helpers import ModelWindowView, StoredWindowLayout, resolve_window_view
from sequifier.io.batch import SequifierInputs
from sequifier.io.preprocessing import normalize_values
from sequifier.io.values import domain_fields, domain_scalar, numeric_scalar
from sequifier.special_tokens import SPECIAL_TOKEN_IDS

PREVIOUS_ACTION = "__policy_previous_action"
DESIRED_RETURN = "__policy_return"


def normalize_policy_features(features, metadata, interface):
    """Apply persisted policy transforms identically in training and prediction."""
    features = dict(features)
    mask = metadata["attention_valid_mask"]
    for column, statistics in interface.selected_columns_statistics.items():
        features[column] = normalize_values(features[column], statistics).masked_fill(
            ~mask, 0.0
        )
    if interface.policy.return_conditioned:
        features[DESIRED_RETURN] = (
            features[DESIRED_RETURN] / interface.policy.returns.scale
        )
    return features


def action_index(value, config: CategoricalPolicyConfig) -> int:
    value = domain_scalar(value, "recorded action")
    try:
        return config.action_values.index(value)
    except ValueError as exc:
        raise ValueError(
            f"action {value!r} is outside action_values {config.action_values!r}"
        ) from exc


def action_global_ids(config: CategoricalPolicyConfig) -> list[int]:
    offset = (
        SPECIAL_TOKEN_IDS.user_start if config.categorical_encoding == "global" else 0
    )
    return list(range(offset, offset + len(config.action_values)))


def action_target(config: CategoricalPolicyConfig) -> str:
    names = config.observation_columns + config.static_columns
    return "__policy_action" if config.action_column in names else config.action_column


def encode_feature(value, column: str, config: CategoricalPolicyConfig):
    mapping = config.id_maps.get(column)
    if mapping is not None:
        try:
            # JSON object keys are strings; preprocessing accepts both forms.
            value = mapping[value] if value in mapping else mapping[str(value)]
        except (KeyError, TypeError) as exc:
            unknown = SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.unknown]
            if unknown not in mapping:
                if (
                    config.categorical_encoding == "global"
                    and SPECIAL_TOKEN_IDS.unknown not in mapping.values()
                ):
                    return SPECIAL_TOKEN_IDS.unknown
                raise ValueError(
                    f"{column}: value {value!r} is outside the fixed vocabulary"
                ) from exc
            value = mapping[unknown]
    count = config.categorical_n_classes.get(column)
    if count is not None:
        domain_scalar(value, column)
        if isinstance(value, str) or value != int(value) or not 0 <= value < count:
            raise ValueError(f"{column}: categorical ID must be in [0, {count})")
        return int(value)
    return numeric_scalar(value, column)


class PolicyWindowView:
    """At decision t, expose o_t and a_(t-1), and supervise a_t separately.

    Previous actions are decoder-local IDs at this boundary. They are converted
    to the persisted input vocabulary here; the start token is input-only.
    """

    def __init__(self, config: CategoricalPolicyConfig):
        self.config = config
        self.view = ModelWindowView(config.context_length, "policy", 0)
        self.resolved = resolve_window_view(
            StoredWindowLayout(config.context_length, 0, 2), self.view
        )

    def episode_inputs(
        self, episode, step: int, returns, action_ids=None
    ) -> SequifierInputs:
        """Expose one causal decision from a validated episode in either lifecycle."""
        start = max(0, step + 1 - self.config.context_length)
        initial = (
            len(self.config.action_values)
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
                len(self.config.action_values)
                if episode.starts_episode
                else action_index(episode.previous_action, self.config)
            )
            if position == 0
            else action_index(episode.actions[position - 1], self.config)
            for position in range(start, step + 2)
        ]
        return self.inputs(observations, previous, None, episode.static_inputs)

    def inputs(
        self,
        observations: Sequence[Mapping],
        previous_actions: Sequence[int],
        desired_returns: Sequence[float] | None,
        static_inputs: Mapping,
    ) -> SequifierInputs:
        config = self.config
        length = len(observations)
        if not length or len(previous_actions) != length:
            raise ValueError(
                "policy observations and previous actions must have matching nonempty lengths"
            )
        if config.return_conditioned:
            if desired_returns is None or len(desired_returns) != length:
                raise ValueError(
                    "return-conditioned windows require one return per decision"
                )
        elif desired_returns is not None:
            raise ValueError("behavior cloning must not consume return inputs")
        start = max(0, length - config.context_length)
        padding = config.context_length - (length - start)
        rows = [
            domain_fields(row, set(config.observation_columns), "policy observation")
            for row in observations[start:]
        ]
        static = domain_fields(
            static_inputs, set(config.static_columns), "policy static inputs"
        )
        features = {}
        for column in config.observation_columns + config.static_columns:
            values = (
                [row[column] for row in rows]
                if column in config.observation_columns
                else [static[column]] * len(rows)
            )
            features[column] = torch.tensor(
                [
                    [0] * padding
                    + [encode_feature(value, column, config) for value in values]
                ],
                dtype=torch.int64
                if column in config.categorical_n_classes
                else torch.float32,
            )
        ids = action_global_ids(config)
        bos = ids[-1] + 1
        preceding = []
        for value in previous_actions[start:]:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= len(ids)
            ):
                raise ValueError(
                    "previous actions require decoder IDs or the explicit start token"
                )
            preceding.append(bos if value == len(ids) else ids[value])
        features[PREVIOUS_ACTION] = torch.tensor(
            [[bos] * padding + preceding], dtype=torch.int64
        )
        if desired_returns is not None:
            features[DESIRED_RETURN] = torch.tensor(
                [
                    [0.0] * padding
                    + [
                        numeric_scalar(value, "desired_return")
                        for value in desired_returns[start:]
                    ]
                ],
                dtype=torch.float32,
            )
        if any(
            value.is_floating_point() and not torch.isfinite(value).all()
            for value in features.values()
        ):
            raise ValueError("policy inputs exceed the finite float32 range")
        metadata = self.resolved.build_masks(torch.tensor([padding], dtype=torch.int64))
        return SequifierInputs(features, metadata)
