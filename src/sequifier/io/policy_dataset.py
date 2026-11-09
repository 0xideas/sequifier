"""Episode-safe policy views over ordinary sequences and imported rollout logs."""

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Sequence

import polars as pl
import torch
from torch.utils.data import Dataset, Sampler

from sequifier.config.policy_config import CategoricalPolicyConfig
from sequifier.io.batch import SequifierBatch
from sequifier.io.episode_corpus import EpisodeCorpus, EpisodeView, episode_content_key
from sequifier.io.partitioning import split_components
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
from sequifier.trajectory.returns import return_to_go


def _return_labels(episode: SequenceEpisode, length: int, gamma: float):
    if episode.transitions is None or episode.transitions.rewards is None:
        raise ValueError("return-conditioned examples require recorded rewards")
    return return_to_go(episode.transitions.rewards[:length], gamma)


class PolicySequenceDataset(Dataset):
    """One decision per standard SequifierBatch; the storage format is incidental."""

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
        self._validated_config = None
        self._validated_statistics = None

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

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
        if self._validated_config == config:
            return
        self.config = config
        self._validated_statistics = None
        # Validate held-out values against the fixed training codec before updates.
        for episode, length in zip(self.episodes, self.lengths):
            for column, values in episode.observations.items():
                for value in values[:length]:
                    encode_feature(value, column, config)
            for column, value in episode.static_inputs.items():
                encode_feature(value, column, config)

        self._validated_config = config.model_copy(deep=True)

    def validate_normalization(self, statistics):
        """Check FP32 transforms episode-by-episode without constructing windows."""
        if self._validated_statistics == statistics:
            return
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

        self._validated_statistics = {k: dict(v) for k, v in statistics.items()}

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
        PolicySequenceDataset(
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
            self.datasets[key] = PolicySequenceDataset(
                EpisodeView.combine([self.corpora[name].view() for name in key]),
                self.config,
                episode_cache_size=self.cache_size,
            )
        return self.datasets[key]
