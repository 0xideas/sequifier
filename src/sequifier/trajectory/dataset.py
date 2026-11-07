"""Episode-split training windows with explicit current-action policy alignment."""

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

from sequifier.config.policy_config import CategoricalPolicyConfig
from sequifier.rl.history import PolicyInputs, action_index, build_policy_inputs
from sequifier.trajectory.returns import labeled_episode_length, return_to_go
from sequifier.trajectory.schema import Trajectory, validate_trajectory
from sequifier.trajectory.store import load_trajectory


@dataclass(frozen=True)
class PolicyExample:
    inputs: PolicyInputs
    actions: torch.Tensor
    action_valid_mask: torch.Tensor
    episode_ids: tuple[str, ...]
    step_ids: tuple[int, ...]


def collate_policy_examples(examples: Sequence[PolicyExample]) -> PolicyExample:
    if not examples:
        raise ValueError("cannot collate an empty policy batch")
    conditioned = examples[0].inputs.desired_returns is not None
    if any(
        (example.inputs.desired_returns is not None) != conditioned
        for example in examples
    ):
        raise ValueError("policy batches cannot mix conditioning modes")
    returns = (
        torch.cat(
            [
                example.inputs.desired_returns
                for example in examples
                if example.inputs.desired_returns is not None
            ]
        )
        if conditioned
        else None
    )
    return PolicyExample(
        inputs=PolicyInputs(
            observations=torch.cat(
                [example.inputs.observations for example in examples]
            ),
            previous_actions=torch.cat(
                [example.inputs.previous_actions for example in examples]
            ),
            attention_valid_mask=torch.cat(
                [example.inputs.attention_valid_mask for example in examples]
            ),
            static_inputs=torch.cat(
                [example.inputs.static_inputs for example in examples]
            ),
            desired_returns=returns,
        ),
        actions=torch.cat([example.actions for example in examples]),
        action_valid_mask=torch.cat(
            [example.action_valid_mask for example in examples]
        ),
        episode_ids=tuple(
            identity for example in examples for identity in example.episode_ids
        ),
        step_ids=tuple(step for example in examples for step in example.step_ids),
    )


class TrajectoryDataset(Dataset):
    """One labeled decision per example; context contains only its causal prefix.

    Context positions have observations and previous actions, while only the
    current (last) position contributes action likelihood. No future observations,
    current actions, or current rewards are arguments to the history builder.
    Return-to-go is an intentional conditioning label computed from rewards.
    """

    def __init__(
        self,
        sources: Sequence[str | Path | Trajectory],
        config: CategoricalPolicyConfig,
        *,
        action_semantics: Literal["policy", "executed_predictions"] = "policy",
    ):
        if action_semantics not in {"policy", "executed_predictions"}:
            raise ValueError(
                "trajectory action_semantics must be policy or executed_predictions"
            )
        self.config = config
        self.action_semantics: Literal["policy", "executed_predictions"] = (
            action_semantics
        )
        self.episodes: list[Trajectory] = []
        self.excluded_episode_ids: list[str] = []
        self.lengths: list[int] = []
        self.returns: list[list[float]] = []
        self.actions: list[list[int]] = []
        self.index: list[tuple[int, int]] = []
        seen = set()
        for source in sources:
            episode = (
                validate_trajectory(source)
                if isinstance(source, Trajectory)
                else load_trajectory(source)
            )
            if episode.episode_id in seen:
                raise ValueError(f"duplicate episode ID {episode.episode_id!r}")
            seen.add(episode.episode_id)
            if (
                episode.format != "sequifier.numeric-policy-trajectory"
                and action_semantics != "executed_predictions"
            ):
                raise ValueError(
                    "forecast sources require explicit action_semantics='executed_predictions'; model forecast alignment is separate"
                )
            if set(episode.seed.observations) != set(config.observation_columns) or set(
                episode.seed.static_inputs
            ) != set(config.static_columns):
                raise ValueError(
                    "trajectory observations/static inputs do not match the policy schema"
                )
            if (
                action_semantics == "policy"
                and sum(episode.seed.attention_valid_mask) != 1
            ):
                raise ValueError(
                    "policy episodes must start with one observation and an explicit previous-action start token"
                )
            if episode.prediction_columns != [config.action_column]:
                raise ValueError(
                    "policy trajectories must contain the configured single action head"
                )
            length = labeled_episode_length(episode, config.returns)
            if length is None:
                self.excluded_episode_ids.append(episode.episode_id)
                continue
            actions = [
                action_index(transition.predictions[config.action_column], config)
                for transition in episode.transitions[:length]
            ]
            returns = return_to_go(
                [transition.reward for transition in episode.transitions[:length]],
                config.returns.gamma,
            )
            episode_index = len(self.episodes)
            self.episodes.append(episode)
            self.lengths.append(length)
            self.actions.append(actions)
            self.returns.append(returns)
            self.index.extend((episode_index, step) for step in range(length))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index: int) -> PolicyExample:
        episode_index, step = self.index[index]
        episode = self.episodes[episode_index]
        inputs = build_policy_inputs(
            self.config,
            [transition.observation for transition in episode.transitions[: step + 1]],
            [len(self.config.action_values)] + self.actions[episode_index][:step],
            self.returns[episode_index][: step + 1]
            if self.config.return_conditioned
            else None,
            episode.seed.static_inputs,
        )
        # Labels have their own mask; they are never passed to policy.forward().
        actions = torch.zeros((1, self.config.context_length), dtype=torch.int64)
        actions[0, -1] = self.actions[episode_index][step]
        valid = torch.zeros_like(actions, dtype=torch.bool)
        valid[0, -1] = True
        return PolicyExample(inputs, actions, valid, (episode.episode_id,), (step,))

    def normalization_rows(self) -> tuple[torch.Tensor, torch.Tensor]:
        observations = [
            [
                transition.observation[column]
                for column in self.config.observation_columns
            ]
            for episode, length in zip(self.episodes, self.lengths)
            for transition in episode.transitions[:length]
        ]
        static = [
            [
                episode.seed.static_inputs[column]
                for column in self.config.static_columns
            ]
            for episode in self.episodes
        ]
        return torch.tensor(observations, dtype=torch.float32), torch.tensor(
            static, dtype=torch.float32
        )


def episode_content_key(episode: Trajectory) -> str:
    """Group repeated demonstrations independently of generated episode IDs."""
    values = episode.model_dump(mode="json")
    values.pop("episode_id")
    values.pop("metadata")
    for transition in values["transitions"]:
        transition.pop("episode_id")
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def split_by_episode(
    dataset: TrajectoryDataset,
    validation_fraction: float = 0.2,
    seed: int = 1010,
    *,
    group_identical: bool = False,
) -> tuple[TrajectoryDataset, TrajectoryDataset]:
    if not 0 < validation_fraction < 1 or len(dataset.episodes) < 2:
        raise ValueError(
            "episode splitting requires at least two eligible episodes and validation_fraction in (0,1)"
        )
    groups = {}
    for episode in dataset.episodes:
        key = episode_content_key(episode) if group_identical else episode.episode_id
        groups.setdefault(key, []).append(episode)
    partitions = list(groups.values())
    if len(partitions) < 2:
        raise ValueError("splitting requires at least two distinct trajectory groups")
    random.Random(seed).shuffle(partitions)
    count = min(len(partitions) - 1, max(1, int(len(partitions) * validation_fraction)))
    validation = [episode for group in partitions[:count] for episode in group]
    training = [episode for group in partitions[count:] for episode in group]
    return (
        TrajectoryDataset(
            training, dataset.config, action_semantics=dataset.action_semantics
        ),
        TrajectoryDataset(
            validation, dataset.config, action_semantics=dataset.action_semantics
        ),
    )
