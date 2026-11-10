"""Episode-safe policy views over ordinary sequences and imported rollout logs."""

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Sequence

import torch
from torch.utils.data import Dataset, Sampler

from sequifier.config.policy_config import CategoricalPolicyConfig
from sequifier.io.batch import SequifierBatch
from sequifier.io.episode_corpus import EpisodeCorpus, EpisodeView, episode_content_key
from sequifier.io.episode_features import EpisodeFeatures, return_labels
from sequifier.io.partitioning import split_components
from sequifier.io.policy_window import (
    PolicyWindowView,
    action_global_ids,
    action_index,
    action_target,
)


class EpisodeDecisions(EpisodeFeatures):
    """Episode-safe decision indexing shared by imitation and transition views."""

    def __init__(
        self,
        sources: Sequence | EpisodeView,
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
        self.demonstration_keys = []
        self.group_ids = []
        self.lengths = []
        self.cumulative_lengths = []
        self._labels = OrderedDict()
        view = (
            sources
            if isinstance(sources, EpisodeView)
            else EpisodeCorpus(sources, config, episode_cache_size).view()
        )
        eligible = []
        seen = set()
        for index in range(len(view)):
            record = view.metadata(index)
            episode_id = record.episode_id
            if episode_id in seen:
                raise ValueError(f"duplicate episode ID {episode_id!r}")
            seen.add(episode_id)
            self.all_episode_ids.append(episode_id)
            length = record.length
            demonstration = record.content_key
            if config.return_conditioned:
                episode = view[index]
                length = episode.return_length(config.returns.finite_horizon)
                if length is None:
                    self.exclusions[episode_id] = (
                        "missing_rewards"
                        if episode.transitions is None
                        or episode.transitions.rewards is None
                        else "unknown_return_tail"
                    )
                    continue
                demonstration = episode_content_key(episode, length=length)
            eligible.append(index)
            self.episode_ids.append(episode_id)
            self.demonstration_keys.append(demonstration)
            self.group_ids.append(record.group_id)
            self.lengths.append(length)
            self.cumulative_lengths.append(
                length + (self.cumulative_lengths[-1] if self.cumulative_lengths else 0)
            )
        self.episodes = view.subset(eligible)
        self.episode_fingerprints = [
            self.episodes.metadata(i).fingerprint for i in range(len(self.episodes))
        ]
        super().__init__(self.episodes, config, self.lengths)

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def sampler(self, generator):
        return EpisodeWindowSampler(self, generator)

    def _episode_labels(self, index: int):
        if index not in self._labels:
            episode = self.episodes[index]
            length = self.lengths[index]
            actions = [
                action_index(value, self.config) for value in episode.actions[:length]
            ]
            returns = (
                return_labels(episode, length, self.config.returns.gamma)
                if self.config.return_conditioned
                else None
            )
            self._labels[index] = actions, returns
            if len(self._labels) > self.episode_cache_size:
                self._labels.popitem(last=False)
        self._labels.move_to_end(index)
        return self._labels[index]

    def locate(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect_right(self.cumulative_lengths, index)
        step = index - (
            self.cumulative_lengths[episode_index - 1] if episode_index else 0
        )
        return episode_index, step


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
    def keys(index):
        result = (
            []
            if dataset.group_ids[index] is None
            else [("group", dataset.group_ids[index])]
        )
        if group_identical:
            result.append(("content", dataset.demonstration_keys[index]))
        return result

    partitions = split_components(
        range(len(dataset.episodes)), keys, validation_fraction, seed
    )
    return tuple(
        type(dataset)(
            dataset.episodes.subset(indices),
            dataset.config,
            episode_cache_size=dataset.episode_cache_size,
        )
        for indices in partitions
    )


class PolicyDataCatalog:
    """Run-owned corpora and reusable decision views for named input sources."""

    def __init__(self, sources, config, cache_size):
        self.sources, self.config, self.cache_size = sources, config, cache_size
        self.corpora = {}
        self.datasets = {}

    def dataset(self, names):
        key = tuple(names)
        if key not in self.datasets:
            for name in key:
                if name not in self.corpora:
                    self.corpora[name] = EpisodeCorpus(
                        [self.sources[name]], self.config, self.cache_size
                    )
            self.datasets[key] = EpisodeDecisions(
                EpisodeView.combine([self.corpora[name].view() for name in key]),
                self.config,
                episode_cache_size=self.cache_size,
            )
        return self.datasets[key]


class PolicySequenceDataset(EpisodeDecisions, Dataset):
    """A supervised batch view over the common episode decision index."""

    def __getitem__(self, index):
        return imitation_example(self, index)


def imitation_example(dataset, index) -> SequifierBatch:
    episode_index, step = dataset.locate(index)
    episode = dataset.episodes[episode_index]
    action_ids, returns = dataset._episode_labels(episode_index)
    inputs = PolicyWindowView(dataset.config).episode_inputs(
        episode, step, returns, action_ids
    )
    if dataset.config.is_joint:
        from sequifier.io.actions import action_batch

        actions = action_batch([action_ids[step]], dataset.config)
        metadata = dict(inputs.metadata)
        metadata["target_valid_mask"] = torch.zeros_like(inputs.attention_valid_mask)
        metadata["target_valid_mask"][0, -1] = True
        return SequifierBatch(
            inputs.inputs,
            {name: value[:, None] for name, value in actions.items()},
            metadata,
        )
    targets = torch.zeros((1, dataset.config.context_length), dtype=torch.int64)
    targets[0, -1] = action_global_ids(dataset.config)[action_ids[step]]
    metadata = dict(inputs.metadata)
    metadata["target_valid_mask"] = torch.zeros_like(inputs.attention_valid_mask)
    metadata["target_valid_mask"][0, -1] = True
    return SequifierBatch(
        inputs.inputs,
        {action_target(dataset.config): targets},
        metadata,
        torch.tensor([episode_index], dtype=torch.int64),
        torch.tensor([episode.item_positions[step]], dtype=torch.int64),
    )
