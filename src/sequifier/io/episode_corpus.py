"""Indexed episode storage and immutable metadata shared by learning views."""

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal

from sequifier.config.policy_train_config import SequenceSourceConfig
from sequifier.fingerprints import fingerprint
from sequifier.io.episode_cache import EpisodeCache
from sequifier.io.policy_window import action_index
from sequifier.io.sequence import SequenceEpisode, register_sequence_name
from sequifier.trajectory.importers import source_episodes
from sequifier.trajectory.sequence_output import sequence_coordinate
from sequifier.trajectory.sources import ResolvedSequenceSource
from sequifier.trajectory.tables import TableEpisodeReference


def _load(source) -> SequenceEpisode:
    if isinstance(source, SequenceEpisode):
        return type(source).model_validate(source.model_dump())
    return source.load()


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_id: str
    sequence_id: int
    group_id: str | None
    length: int
    content_key: str
    fingerprint: str


@dataclass(frozen=True)
class EpisodeRecords:
    """Indexed source readers and metadata for one episode interpretation."""

    sources: tuple
    metadata: tuple[EpisodeMetadata, ...]


class EpisodeCorpus:
    """Own one bounded decoded cache over immutable source records."""

    def __init__(
        self,
        sources,
        config,
        cache_size=8,
        *,
        cache_owner=None,
        records=None,
        sequence_names=None,
    ):
        if cache_size < 1:
            raise ValueError("episode_cache_size must be positive")
        self.cache = (
            cache_owner.episode_cache(cache_size)
            if cache_owner is not None
            else EpisodeCache(cache_size)
        )
        self.records = records if records is not None else self._index(sources, config)
        names = {} if sequence_names is None else sequence_names
        for record in self.records.metadata:
            register_sequence_name(names, record.sequence_id, record.episode_id)

    def _index(self, sources, config):
        sources = tuple(
            episode
            for source in sources
            for episode in (
                source_episodes(source, config)
                if isinstance(source, (SequenceSourceConfig, ResolvedSequenceSource))
                else [source]
            )
        )
        metadata = []
        seen = set()
        for index in range(len(sources)):
            episode = self.cache.get(index, lambda i: _load(sources[i]))
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
            metadata.append(
                EpisodeMetadata(
                    episode.episode_id,
                    episode.sequence_id
                    if episode.sequence_id is not None
                    else sequence_coordinate(episode.episode_id),
                    episode.group_id,
                    len(episode.actions),
                    getattr(sources[index], "content_key", None)
                    or episode_content_key(episode),
                    fingerprint(
                        episode.model_dump(mode="json", exclude={"sequence_id"})
                    ),
                )
            )
        return EpisodeRecords(sources, tuple(metadata))

    def __len__(self):
        return len(self.records.sources)

    def __getitem__(self, index):
        return self.cache.get(index, self._load_episode)

    def _load_episode(self, index):
        return _load(self.records.sources[index])

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
        return corpus.records.metadata[position]

    def source(self, index):
        corpus, position = self.entries[index]
        return corpus.records.sources[position]

    def used_for_fitting(self, index):
        corpus, position = self.entries[index]
        source = corpus.records.sources[position]
        return isinstance(source, TableEpisodeReference) and source.used_for_fitting

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
