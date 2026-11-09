"""Episode-safe policy views over ordinary sequences and imported rollout logs."""

import hashlib
import json
import random
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from decimal import Decimal

import polars as pl
import torch
from torch.utils.data import Dataset, Sampler

from sequifier.config.policy_config import CategoricalPolicyConfig
from sequifier.config.policy_train_config import SequenceSourceConfig
from sequifier.io.batch import SequifierBatch
from sequifier.io.policy_window import (
    PolicyWindowView,
    action_global_ids,
    action_index,
    action_target,
    encode_feature,
)
from sequifier.io.preprocessing import (
    _finite_mean_and_std,
    create_id_map,
    get_combined_statistics,
    normalize_values,
)
from sequifier.io.sequence import SequenceEpisode
from sequifier.io.values import numeric_scalar
from sequifier.trajectory.importers import source_episodes
from sequifier.trajectory.returns import return_to_go


def _load(source) -> SequenceEpisode:
    if isinstance(source, SequenceEpisode):
        return SequenceEpisode.model_validate(source.model_dump())
    return source.load()


class EpisodeCollection:
    """Keep source-backed episodes on disk, with a bounded per-worker cache."""

    def __init__(self, sources, cache_size):
        self.sources = list(sources)
        self.cache_size = cache_size
        self.cache: OrderedDict[int, SequenceEpisode] = OrderedDict()

    def __len__(self):
        return len(self.sources)

    def __iter__(self) -> Iterator[SequenceEpisode]:
        return (self[index] for index in range(len(self)))

    def __getitem__(self, index: int) -> SequenceEpisode:
        if index not in self.cache:
            self.cache[index] = _load(self.sources[index])
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


def _canonical(value):
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ["number", str(Decimal(str(value)).normalize())]
    return value


def episode_content_key(episode, *, length=None, returns=None) -> str:
    """Hash learning content, independently of log format, IDs and provenance."""
    length = len(episode.actions) if length is None else length
    values = {
        "observations": {
            column: entries[:length] for column, entries in episode.observations.items()
        },
        "actions": episode.actions[:length],
        "static_inputs": episode.static_inputs,
        "previous_action": episode.previous_action,
        "starts_episode": episode.starts_episode,
        "returns": returns,
    }
    return hashlib.sha256(
        json.dumps(_canonical(values), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _return_labels(episode: SequenceEpisode, length: int, gamma: float):
    if episode.transitions is None or episode.transitions.rewards is None:
        raise ValueError("return-conditioned examples require recorded rewards")
    return return_to_go(episode.transitions.rewards[:length], gamma)


class PolicySequenceDataset(Dataset):
    """One decision per standard SequifierBatch; the storage format is incidental."""

    def __init__(
        self,
        sources: Sequence,
        config: CategoricalPolicyConfig,
        *,
        episode_cache_size: int = 8,
    ):
        if episode_cache_size < 1:
            raise ValueError("episode_cache_size must be positive")
        self.config = config
        self.episode_cache_size = episode_cache_size
        self.exclusions = {}
        self.all_episode_ids = []
        self.episode_ids = []
        self.content_keys = []
        self.demonstration_keys = []
        self.group_ids = []
        self.lengths = []
        self.cumulative_lengths = []
        self._labels = OrderedDict()
        eligible = []
        seen = set()
        expanded = [
            episode
            for source in sources
            for episode in (
                source_episodes(source, config)
                if isinstance(source, SequenceSourceConfig)
                else [source]
            )
        ]
        categorical = (
            set(config.categorical_columns) | config.categorical_n_classes.keys()
        )
        for source in expanded:
            episode = _load(source)
            if episode.episode_id in seen:
                raise ValueError(f"duplicate episode ID {episode.episode_id!r}")
            seen.add(episode.episode_id)
            self.all_episode_ids.append(episode.episode_id)
            if set(episode.observations) != set(config.observation_columns) or set(
                episode.static_inputs
            ) != set(config.static_columns):
                raise ValueError(
                    "sequence observations/static inputs do not match the policy schema"
                )
            # Validate all labels, even in an excluded unknown return tail.
            for value in episode.actions:
                action_index(value, config)
            if episode.previous_action is not None:
                action_index(episode.previous_action, config)
            for column, values in episode.observations.items():
                for value in values:
                    if column not in categorical:
                        numeric_scalar(value, column)
            for column, value in episode.static_inputs.items():
                if column not in categorical:
                    numeric_scalar(value, column)
            length = (
                episode.return_length(config.returns.finite_horizon)
                if config.return_conditioned
                else len(episode.actions)
            )
            if length is None:
                self.exclusions[episode.episode_id] = (
                    "missing_rewards"
                    if episode.transitions is None
                    or episode.transitions.rewards is None
                    else "unknown_return_tail"
                )
                continue
            returns = (
                _return_labels(episode, length, config.returns.gamma)
                if config.return_conditioned
                else None
            )
            eligible.append(source)
            self.episode_ids.append(episode.episode_id)
            self.content_keys.append(
                episode_content_key(episode, length=length, returns=returns)
            )
            self.demonstration_keys.append(episode_content_key(episode, length=length))
            self.group_ids.append(episode.group_id)
            self.lengths.append(length)
            self.cumulative_lengths.append(
                length + (self.cumulative_lengths[-1] if self.cumulative_lengths else 0)
            )
        self.episodes = EpisodeCollection(eligible, episode_cache_size)

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    @property
    def data_contract(self):
        return hashlib.sha256(
            json.dumps([self.content_keys, self.episode_ids]).encode()
        ).hexdigest()

    def sampler(self, generator):
        return EpisodeWindowSampler(self, generator)

    def fit_categorical_config(self) -> CategoricalPolicyConfig:
        """Fit only declared raw categorical features, using the training episodes."""
        values = self.config.model_dump()
        for column in self.config.categorical_columns:
            if (
                column in self.config.id_maps
                or column in self.config.categorical_n_classes
            ):
                continue
            vocabulary = set()
            for episode, length in zip(self.episodes, self.lengths):
                vocabulary.update(
                    episode.observations[column][:length]
                    if column in episode.observations
                    else [episode.static_inputs[column]]
                )
            if not vocabulary:
                raise ValueError(f"cannot fit empty categorical column {column!r}")
            if len({isinstance(value, str) for value in vocabulary}) > 1:
                raise ValueError(
                    f"{column}: mixed string/numeric category labels are ambiguous"
                )
            if any(
                not isinstance(value, str) and value != int(value)
                for value in vocabulary
            ):
                raise ValueError(
                    f"{column}: categorical numeric labels must be integers"
                )
            mapping = create_id_map(pl.DataFrame({column: sorted(vocabulary)}), column)
            values["id_maps"][column] = mapping
            key = (
                "observation_n_classes"
                if column in self.config.observation_columns
                else "static_n_classes"
            )
            values[key][column] = max(mapping.values()) + 1
        return CategoricalPolicyConfig.model_validate(values)

    def use_config(self, config: CategoricalPolicyConfig) -> None:
        self.config = config
        # Validate held-out values against the fixed training codec before updates.
        for episode, length in zip(self.episodes, self.lengths):
            for column, values in episode.observations.items():
                for value in values[:length]:
                    encode_feature(value, column, config)
            for column, value in episode.static_inputs.items():
                encode_feature(value, column, config)

    def validate_normalization(self, statistics):
        """Check FP32 transforms episode-by-episode without constructing windows."""
        for index, (episode, length) in enumerate(zip(self.episodes, self.lengths)):
            for column in self.config.observation_columns + self.config.static_columns:
                if column in self.config.categorical_n_classes:
                    continue
                values = (
                    episode.observations[column][:length]
                    if column in episode.observations
                    else [episode.static_inputs[column]]
                )
                tensor = torch.tensor(values, dtype=torch.float32)
                transformed = normalize_values(tensor, statistics[column])
                if (
                    not torch.isfinite(tensor).all()
                    or not torch.isfinite(transformed).all()
                ):
                    raise ValueError(
                        f"episode {episode.episode_id}, {column}: non-finite FP32 input/normalization"
                    )
            if self.config.return_conditioned:
                _, returns = self._episode_labels(index)
                if not torch.isfinite(
                    torch.tensor(returns, dtype=torch.float32)
                    / self.config.returns.scale
                ).all():
                    raise ValueError(
                        f"episode {episode.episode_id}: non-finite conditioned return"
                    )

    def _episode_labels(self, index: int):
        if index not in self._labels:
            episode = self.episodes[index]
            length = self.lengths[index]
            actions = [
                action_index(value, self.config) for value in episode.actions[:length]
            ]
            returns = (
                _return_labels(episode, length, self.config.returns.gamma)
                if self.config.return_conditioned
                else None
            )
            self._labels[index] = actions, returns
            if len(self._labels) > self.episode_cache_size:
                self._labels.popitem(last=False)
        self._labels.move_to_end(index)
        return self._labels[index]

    def __getitem__(self, index: int) -> SequifierBatch:
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
        inputs = PolicyWindowView(self.config).episode_inputs(
            episode, step, returns, action_ids
        )
        targets = torch.zeros((1, self.config.context_length), dtype=torch.int64)
        targets[0, -1] = action_global_ids(self.config)[action_ids[step]]
        metadata = dict(inputs.metadata)
        metadata["target_valid_mask"] = torch.zeros_like(inputs.attention_valid_mask)
        metadata["target_valid_mask"][0, -1] = True
        return SequifierBatch(
            inputs.inputs,
            {action_target(self.config): targets},
            metadata,
            torch.tensor([episode_index], dtype=torch.int64),
            torch.tensor([episode.item_positions[step]], dtype=torch.int64),
        )

    def normalization_statistics(self):
        """Use the same stable sample moments as supervised preprocessing."""
        moments = {}
        columns = self.config.observation_columns + self.config.static_columns
        for episode, length in zip(self.episodes, self.lengths):
            for column in columns:
                if column in self.config.categorical_n_classes:
                    continue
                values = (
                    episode.observations[column][:length]
                    if column in episode.observations
                    else [episode.static_inputs[column]]
                )
                mean, std = _finite_mean_and_std(pl.Series(column, values), column)
                size = len(values)
                if column in moments:
                    count, old_mean, old_std = moments[column]
                    mean, std = get_combined_statistics(
                        size, mean, std, count, old_mean, old_std
                    )
                    size += count
                moments[column] = size, mean, std
        return {
            column: {"mean": mean, "std": std}
            for column, (_, mean, std) in moments.items()
        }


class EpisodeWindowSampler(Sampler):
    """Shuffle episodes and their decisions while keeping disk access local."""

    def __init__(self, dataset, generator):
        self.dataset, self.generator = dataset, generator

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


def split_by_episode(
    dataset, validation_fraction=0.2, seed=1010, *, group_identical=False
):
    if not 0 < validation_fraction < 1 or len(dataset.episodes) < 2:
        raise ValueError(
            "episode splitting requires at least two eligible episodes and validation_fraction in (0,1)"
        )
    # Union both collection groups and repeated content, including transitive ties.
    parents = list(range(len(dataset.episodes)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index in range(len(parents)):
        keys = (
            [("group", dataset.group_ids[index])]
            if dataset.group_ids[index] is not None
            else []
        )
        if group_identical:
            keys.append(("content", dataset.demonstration_keys[index]))
        for key in keys:
            if key in seen:
                parents[root(index)] = root(seen[key])
            seen[key] = index
    groups = {}
    for index, source in enumerate(dataset.episodes.sources):
        groups.setdefault(root(index), []).append(source)
    partitions = list(groups.values())
    if len(partitions) < 2:
        raise ValueError("splitting requires at least two independent episode groups")
    random.Random(seed).shuffle(partitions)
    count = min(len(partitions) - 1, max(1, int(len(partitions) * validation_fraction)))
    return tuple(
        PolicySequenceDataset(
            [source for group in partition for source in group],
            dataset.config,
            episode_cache_size=dataset.episode_cache_size,
        )
        for partition in (partitions[count:], partitions[:count])
    )
