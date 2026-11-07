"""Episode-split training windows with explicit current-action policy alignment."""

import hashlib
import json
import random
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

import torch
from torch.utils.data import Dataset, Sampler

from sequifier.config.policy_config import CategoricalPolicyConfig
from sequifier.rl.history import (
    PolicyInputs,
    action_index,
    build_policy_inputs,
    collate_policy_inputs,
)
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
    return PolicyExample(
        inputs=collate_policy_inputs([example.inputs for example in examples]),
        actions=torch.cat([example.actions for example in examples]),
        action_valid_mask=torch.cat(
            [example.action_valid_mask for example in examples]
        ),
        episode_ids=tuple(
            identity for example in examples for identity in example.episode_ids
        ),
        step_ids=tuple(step for example in examples for step in example.step_ids),
    )


class EpisodeCollection(Sequence[Trajectory]):
    """Keep path-backed episodes on disk, with a bounded per-worker cache."""

    def __init__(self, sources, cache_size: int):
        self.sources = list(sources)
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.sources)

    @overload
    def __getitem__(self, index: int) -> Trajectory: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Trajectory]: ...

    def __getitem__(self, index: int | slice) -> Trajectory | Sequence[Trajectory]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        source = self.sources[index]
        if isinstance(source, Trajectory):
            return source
        if index not in self.cache:
            self.cache[index] = load_trajectory(source)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


class TrajectoryDataset(Dataset):
    """One labeled decision per example with a bounded causal context.

    File-backed corpora retain only episode metadata and a small episode cache.
    Return labels are cached per episode, never reconstructed for every prefix.
    """

    def __init__(
        self,
        sources: Sequence[str | Path | Trajectory],
        config: CategoricalPolicyConfig,
        *,
        action_semantics: Literal["policy", "executed_predictions"] = "policy",
        episode_cache_size: int = 8,
    ):
        if action_semantics not in {"policy", "executed_predictions"}:
            raise ValueError(
                "trajectory action_semantics must be policy or executed_predictions"
            )
        if episode_cache_size < 1:
            raise ValueError("episode_cache_size must be positive")
        self.config = config
        self.action_semantics: Literal["policy", "executed_predictions"] = (
            action_semantics
        )
        self.excluded_episode_ids = []
        self.all_episode_ids = []
        self.episode_ids = []
        self.content_keys = []
        self.lengths = []
        self.cumulative_lengths = []
        self.episode_cache_size = episode_cache_size
        self._labels = OrderedDict()
        eligible = []
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
            self.all_episode_ids.append(episode.episode_id)
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
            length = (
                labeled_episode_length(episode, config.returns)
                if config.return_conditioned
                else len(episode.transitions)
            )
            if length is None:
                self.excluded_episode_ids.append(episode.episode_id)
                continue
            for transition in episode.transitions[:length]:
                action_index(transition.predictions[config.action_column], config)
            eligible.append(source)
            self.episode_ids.append(episode.episode_id)
            self.content_keys.append(episode_content_key(episode))
            self.lengths.append(length)
            self.cumulative_lengths.append(
                length + (self.cumulative_lengths[-1] if self.cumulative_lengths else 0)
            )
        self.episodes = EpisodeCollection(eligible, episode_cache_size)

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def _episode_labels(self, index: int):
        if index not in self._labels:
            transitions = self.episodes[index].transitions[: self.lengths[index]]
            actions = [
                action_index(
                    transition.predictions[self.config.action_column], self.config
                )
                for transition in transitions
            ]
            returns = (
                return_to_go(
                    [transition.reward for transition in transitions],
                    self.config.returns.gamma,
                )
                if self.config.return_conditioned
                else None
            )
            self._labels[index] = actions, returns
            if len(self._labels) > self.episode_cache_size:
                self._labels.popitem(last=False)
        self._labels.move_to_end(index)
        return self._labels[index]

    def __getitem__(self, index: int) -> PolicyExample:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect_right(self.cumulative_lengths, index)
        step = index - (
            self.cumulative_lengths[episode_index - 1] if episode_index else 0
        )
        episode = self.episodes[episode_index]
        action_ids, returns = self._episode_labels(episode_index)
        start = max(0, step + 1 - self.config.context_length)
        previous = (
            [len(self.config.action_values)] if start == 0 else []
        ) + action_ids[max(0, start - 1) : step]
        inputs = build_policy_inputs(
            self.config,
            [
                transition.observation
                for transition in episode.transitions[start : step + 1]
            ],
            previous,
            returns[start : step + 1] if returns is not None else None,
            episode.seed.static_inputs,
        )
        actions = torch.zeros((1, self.config.context_length), dtype=torch.int64)
        actions[0, -1] = action_ids[step]
        valid = torch.zeros_like(actions, dtype=torch.bool)
        valid[0, -1] = True
        return PolicyExample(inputs, actions, valid, (episode.episode_id,), (step,))

    def normalization_statistics(self) -> dict[str, dict[str, float]]:
        """Combine episode moments without materializing all training rows."""
        columns = self.config.observation_columns + self.config.static_columns
        moments = {
            column: (0, 0.0, 0.0)
            for column in columns
            if column not in self.config.observation_n_classes
        }
        for episode, length in zip(self.episodes, self.lengths):
            for column in moments:
                values = (
                    [
                        transition.observation[column]
                        for transition in episode.transitions[:length]
                    ]
                    if column in self.config.observation_columns
                    else [episode.seed.static_inputs[column]]
                )
                fitted = torch.tensor(values, dtype=torch.float64)
                count, mean, m2 = moments[column]
                size = len(values)
                batch_mean = float(fitted.mean())
                delta = batch_mean - mean
                total = count + size
                moments[column] = (
                    total,
                    mean + delta * size / total,
                    m2
                    + float(((fitted - batch_mean) ** 2).sum())
                    + delta**2 * count * size / total,
                )
        return {
            column: {"mean": mean, "std": max((m2 / count) ** 0.5, 1e-6)}
            for column, (count, mean, m2) in moments.items()
        }

    def normalization_rows(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialized convenience API retained for Python callers."""
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


class EpisodeWindowSampler(Sampler):
    """Shuffle episodes and their decisions while keeping disk access local."""

    def __init__(self, dataset: TrajectoryDataset, generator: torch.Generator):
        self.dataset = dataset
        self.generator = generator

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for episode in torch.randperm(
            len(self.dataset.lengths), generator=self.generator
        ).tolist():
            offset = self.dataset.cumulative_lengths[episode - 1] if episode else 0
            for step in torch.randperm(
                self.dataset.lengths[episode], generator=self.generator
            ).tolist():
                yield offset + step


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
    for index, source in enumerate(dataset.episodes.sources):
        key = (
            dataset.content_keys[index]
            if group_identical
            else dataset.episode_ids[index]
        )
        groups.setdefault(key, []).append(source)
    partitions = list(groups.values())
    if len(partitions) < 2:
        raise ValueError("splitting requires at least two distinct trajectory groups")
    random.Random(seed).shuffle(partitions)
    count = min(len(partitions) - 1, max(1, int(len(partitions) * validation_fraction)))
    validation = [episode for group in partitions[:count] for episode in group]
    training = [episode for group in partitions[count:] for episode in group]
    return (
        TrajectoryDataset(
            training,
            dataset.config,
            action_semantics=dataset.action_semantics,
            episode_cache_size=dataset.episode_cache_size,
        ),
        TrajectoryDataset(
            validation,
            dataset.config,
            action_semantics=dataset.action_semantics,
            episode_cache_size=dataset.episode_cache_size,
        ),
    )
