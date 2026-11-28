"""Episode-safe policy views over ordinary sequences and imported rollout logs."""

from bisect import bisect_right
from collections.abc import Sequence

import torch
from torch.utils.data import Dataset, Sampler

from sequifier.config.policy_inputs import PolicyEpisodeSpec
from sequifier.io.batch import (
    ModelEncodedPolicyBatch,
    ModelEncodedPolicyInputs,
    SequifierBatch,
)
from sequifier.io.episode_cache import EpisodeCache
from sequifier.io.episode_corpus import EpisodeCorpus, EpisodeView, episode_content_key
from sequifier.io.episode_features import EpisodeFeatures, return_labels
from sequifier.io.partitioning import sequence_assignments
from sequifier.io.policy_window import PolicyWindowView, action_index


class EpisodeDecisions(EpisodeFeatures):
    """Episode-safe decision indexing shared by imitation and transition views."""

    def __init__(
        self,
        sources: Sequence | EpisodeView,
        config: PolicyEpisodeSpec,
        *,
        episode_cache_size: int = 8,
        cache_owner=None,
    ):
        if episode_cache_size < 1:
            raise ValueError("episode_cache_size must be positive")
        self.config = config
        self.episode_cache_size = episode_cache_size
        self.exclusions = {}
        self.all_episode_ids = []
        self.episode_ids = []
        self.sequence_ids = []
        self.demonstration_keys = []
        self.group_ids = []
        self.lengths = []
        self.cumulative_lengths = []
        self.cache_owner = cache_owner
        self._labels = (
            cache_owner.episode_cache(episode_cache_size)
            if cache_owner is not None
            else EpisodeCache(episode_cache_size)
        )
        view = (
            sources
            if isinstance(sources, EpisodeView)
            else EpisodeCorpus(
                sources, config, episode_cache_size, cache_owner=cache_owner
            ).view()
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
            self.sequence_ids.append(record.sequence_id)
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

    def subset(self, indices):
        return type(self)(
            self.episodes.subset(indices),
            self.config,
            episode_cache_size=self.episode_cache_size,
            cache_owner=self.cache_owner,
        )

    def _episode_labels(self, index: int):
        return self._labels.get(index, self._load_labels)

    def _load_labels(self, index):
        episode = self.episodes[index]
        length = self.lengths[index]
        actions = [
            action_index(episode.decision(step).action, self.config)
            for step in range(length)
        ]
        returns = (
            return_labels(episode, length, self.config.returns.gamma)
            if self.config.return_conditioned
            else None
        )
        return actions, returns

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


def episode_split_assignments(dataset, fractions, seed, *, group_identical):
    """Return one supervised split index per eligible policy episode."""
    sequences = dataset.sequence_ids
    content = {}
    if group_identical and len(fractions) > 1:
        for index, seq in enumerate(sequences):
            reference = dataset.episodes.source(index)
            content[seq] = (
                getattr(reference, "split_content_key", None)
                or dataset.demonstration_keys[index]
            )
    assignments = sequence_assignments(
        sequences,
        dict(zip(sequences, dataset.group_ids, strict=True)),
        content,
        fractions,
        seed,
    )
    if group_identical:
        # Distinct source records can become the same demonstration after selecting
        # model features. Keep the physical partition authoritative and catch leakage.
        owners = {}
        for seq, key in zip(sequences, dataset.demonstration_keys, strict=True):
            if owners.setdefault(key, assignments[seq]) != assignments[seq]:
                raise ValueError(
                    "Identical policy demonstrations cross recorded splits. "
                    "Give these sequences the same groupId in the source."
                )
    return [assignments[seq] for seq in sequences]


def episode_partition_indices(
    dataset, validation_fraction=0.2, seed=1010, *, group_identical=False
):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    assignments = episode_split_assignments(
        dataset,
        [1 - validation_fraction, validation_fraction],
        seed,
        group_identical=group_identical,
    )
    return tuple(
        tuple(i for i, part in enumerate(assignments) if part == split)
        for split in range(2)
    )


def split_by_episode(
    dataset, validation_fraction=0.2, seed=1010, *, group_identical=False
):
    return tuple(
        dataset.subset(indices)
        for indices in episode_partition_indices(
            dataset, validation_fraction, seed, group_identical=group_identical
        )
    )


class PolicyDataCatalog:
    """Run-owned corpora and reusable decision views for named input sources."""

    def __init__(
        self,
        sources,
        config,
        cache_size,
        *,
        preference_manifests=None,
        episode_records=None,
        sequence_names=None,
        cache_owner=None,
    ):
        self.sources, self.config, self.cache_size = sources, config, cache_size
        self.cache_owner = cache_owner
        self.corpora = {}
        self.sequence_names = {} if sequence_names is None else sequence_names
        self.episode_records = episode_records if episode_records is not None else {}
        self.datasets = {}
        self.preference_manifests = (
            preference_manifests if preference_manifests is not None else {}
        )

    def preference_pairs(self, path):
        if path not in self.preference_manifests:
            from sequifier.trajectory.preferences import load_preferences

            self.preference_manifests[path] = load_preferences(path).pairs
        return self.preference_manifests[path]

    def dataset(self, names):
        key = tuple(names)
        if key not in self.datasets:
            for name in key:
                if name not in self.corpora:
                    corpus = EpisodeCorpus(
                        [self.sources[name]],
                        self.config,
                        self.cache_size,
                        cache_owner=self.cache_owner,
                        records=self.episode_records.get(name),
                        sequence_names=self.sequence_names,
                    )
                    self.corpora[name] = corpus
                    self.episode_records[name] = corpus.records
            self.datasets[key] = EpisodeDecisions(
                EpisodeView.combine([self.corpora[name].view() for name in key]),
                self.config,
                episode_cache_size=self.cache_size,
                cache_owner=self.cache_owner,
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
    inputs = PolicyWindowView(dataset.config).episode_inputs(episode, step, returns)
    from sequifier.io.actions import action_batch

    actions = action_batch([action_ids[step]], dataset.config)
    metadata = dict(inputs.metadata)
    metadata["target_valid_mask"] = torch.zeros_like(inputs.attention_valid_mask)
    metadata["target_valid_mask"][0, -1] = True
    batch_type = (
        ModelEncodedPolicyBatch
        if isinstance(inputs, ModelEncodedPolicyInputs)
        else SequifierBatch
    )
    return batch_type(
        inputs.inputs,
        {name: value[:, None] for name, value in actions.items()},
        metadata,
    )
