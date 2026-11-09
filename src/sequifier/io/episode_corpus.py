"""Indexed episode storage and immutable metadata shared by learning views."""

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal

from sequifier.config.policy_train_config import SequenceSourceConfig
from sequifier.fingerprints import fingerprint
from sequifier.io.policy_window import action_index
from sequifier.io.sequence import SequenceEpisode
from sequifier.io.values import numeric_scalar
from sequifier.trajectory.importers import source_episodes


def _load(source) -> SequenceEpisode:
    if isinstance(source, SequenceEpisode):
        return SequenceEpisode.model_validate(source.model_dump())
    return source.load()


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_id: str
    group_id: str | None
    length: int
    content_key: str
    fingerprint: str


class EpisodeCorpus:
    """Own validated episode metadata and one bounded storage cache."""

    def __init__(self, sources, config, cache_size=8):
        if cache_size < 1:
            raise ValueError("episode_cache_size must be positive")
        self.sources = [
            episode
            for source in sources
            for episode in (
                source_episodes(source, config)
                if isinstance(source, SequenceSourceConfig)
                else [source]
            )
        ]
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.metadata = []
        categorical = (
            set(config.categorical_columns) | config.categorical_n_classes.keys()
        )
        seen = set()
        for index in range(len(self.sources)):
            episode = self[index]
            if episode.episode_id in seen:
                raise ValueError(f"duplicate episode ID {episode.episode_id!r}")
            seen.add(episode.episode_id)
            if set(episode.observations) != set(config.observation_columns) or set(
                episode.static_inputs
            ) != set(config.static_columns):
                raise ValueError(
                    "sequence observations/static inputs do not match the policy schema"
                )
            for value in episode.actions:
                action_index(value, config)
            if episode.previous_action is not None:
                action_index(episode.previous_action, config)
            for column, values in episode.observations.items():
                if column not in categorical:
                    for value in values:
                        numeric_scalar(value, column)
            for column, value in episode.static_inputs.items():
                if column not in categorical:
                    numeric_scalar(value, column)
            self.metadata.append(
                EpisodeMetadata(
                    episode.episode_id,
                    episode.group_id,
                    len(episode.actions),
                    episode_content_key(episode),
                    fingerprint(episode.model_dump(mode="json")),
                )
            )

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        if index not in self.cache:
            self.cache[index] = _load(self.sources[index])
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def view(self):
        return EpisodeView([(self, i) for i in range(len(self))])


class EpisodeView:
    """A lightweight ordered partition; storage and metadata remain corpus-owned."""

    def __init__(self, entries):
        self.entries = tuple(entries)

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, index):
        corpus, position = self.entries[index]
        return corpus[position]

    def metadata(self, index):
        corpus, position = self.entries[index]
        return corpus.metadata[position]

    def subset(self, indices):
        return EpisodeView([self.entries[i] for i in indices])

    @classmethod
    def combine(cls, views):
        return cls([entry for view in views for entry in view.entries])


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
