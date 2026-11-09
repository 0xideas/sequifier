"""Lazy preference storage and leakage-safe partition views."""

import hashlib
from dataclasses import dataclass

import torch

from sequifier.fingerprints import fingerprint
from sequifier.io.episode_corpus import EpisodeView, episode_content_key
from sequifier.io.partitioning import split_components
from sequifier.io.policy_dataset import PolicySequenceDataset
from sequifier.io.policy_window import PolicyWindowView, action_index
from sequifier.trajectory.preferences import load_preferences


@dataclass(frozen=True)
class PreferenceData:
    training: "PreferenceDataset"
    validation: "PreferenceDataset"

    @property
    def contract(self):
        return [self.training.contract, self.validation.contract]


class PreferenceDataset:
    def __init__(self, pairs, corpora, config):
        self.pairs, self.corpora, self.config = tuple(pairs), corpora, config
        self.lookup = {
            name: {episode: i for i, episode in enumerate(data.episode_ids)}
            for name, data in corpora.items()
        }
        self._fitting = None
        self._validated_config = None
        self.contract = []
        for pair in self.pairs:
            entries = []
            for span in (pair.chosen, pair.rejected):
                episode = self.episode(span)
                if span.end > len(episode.actions):
                    raise ValueError(
                        f"pair {pair.pair_id}: span exceeds episode {span.episode_id}"
                    )
                entries.append(
                    {"span": span.model_dump(), "content": episode_content_key(episode)}
                )
            if self.alternative_key(pair.chosen) == self.alternative_key(pair.rejected):
                raise ValueError(
                    f"pair {pair.pair_id}: chosen/rejected alternatives have identical decision content"
                )
            if (
                pair.chosen.source == pair.rejected.source
                and pair.chosen.episode_id == pair.rejected.episode_id
                and max(pair.chosen.start, pair.rejected.start)
                < min(pair.chosen.end, pair.rejected.end)
            ):
                raise ValueError(f"pair {pair.pair_id}: chosen/rejected spans overlap")
            self.contract.append({"pair": pair.model_dump(), "alternatives": entries})

    def __len__(self):
        return len(self.pairs)

    def episode(self, span):
        try:
            return self.corpora[span.source].episodes[
                self.lookup[span.source][span.episode_id]
            ]
        except KeyError as exc:
            raise ValueError(
                f"preference references unknown source/episode {span.source}:{span.episode_id}"
            ) from exc

    def alternative_key(self, span):
        episode = self.episode(span)
        start = max(0, span.start + 1 - self.config.context_length)
        return fingerprint(
            {
                "observations": {
                    k: v[start : span.end] for k, v in episode.observations.items()
                },
                "actions": episode.actions[max(0, start - 1) : span.end],
                "static": episode.static_inputs,
                "starts_episode": episode.starts_episode,
                "previous_action": episode.previous_action,
            }
        )

    def components(self, pair):
        keys = [("context", pair.context_id)]
        if pair.group_id is not None:
            keys.append(("group", pair.group_id))
        for span in (pair.chosen, pair.rejected):
            episode = self.episode(span)
            keys += [
                ("episode", span.episode_id),
                ("content", episode_content_key(episode)),
                ("alternative", self.alternative_key(span)),
            ]
            if episode.group_id is not None:
                keys.append(("group", episode.group_id))
        return keys

    def fitting_dataset(self):
        if self._fitting is None:
            entries = {}
            for pair in self.pairs:
                for span in (pair.chosen, pair.rejected):
                    index = self.lookup[span.source][span.episode_id]
                    entries[(span.source, span.episode_id)] = self.corpora[
                        span.source
                    ].episodes.entries[index]
            cache_size = next(iter(self.corpora.values())).episode_cache_size
            self._fitting = PolicySequenceDataset(
                EpisodeView(entries.values()),
                self.config,
                episode_cache_size=cache_size,
            )
        return self._fitting

    def use_config(self, config):
        if self._validated_config == config:
            return
        self.config = config
        view = PolicyWindowView(config)
        contexts = {}
        for pair in self.pairs:
            initial = []
            alternatives = []
            for span in (pair.chosen, pair.rejected):
                episode = self.episode(span)
                digest = hashlib.sha256()
                # Validate every selected decision and all its causal inputs before training.
                for step in range(span.start, span.end):
                    inputs = view.episode_inputs(episode, step, None)
                    action = action_index(episode.actions[step], config)
                    digest.update(
                        fingerprint(
                            {
                                "features": {
                                    k: v.tolist() for k, v in inputs.inputs.items()
                                },
                                "masks": {
                                    k: v.tolist() for k, v in inputs.metadata.items()
                                },
                                "action": action,
                            }
                        ).encode()
                    )
                    if step == span.start:
                        initial.append(inputs)
                alternatives.append(digest.hexdigest())
            if alternatives[0] == alternatives[1]:
                raise ValueError(
                    f"pair {pair.pair_id}: alternatives have identical canonical executable decisions"
                )
            context_key = fingerprint(
                {
                    "features": {k: v.tolist() for k, v in initial[0].inputs.items()},
                    "masks": {k: v.tolist() for k, v in initial[0].metadata.items()},
                }
            )
            if pair.context_id in contexts and contexts[pair.context_id] != context_key:
                raise ValueError(
                    f"context {pair.context_id}: ID refers to different canonical initial histories"
                )
            contexts[pair.context_id] = context_key
            left, right = initial
            for mapping in ("inputs", "metadata"):
                lmap, rmap = getattr(left, mapping), getattr(right, mapping)
                if lmap.keys() != rmap.keys() or any(
                    not torch.equal(lmap[k], rmap[k]) for k in lmap
                ):
                    raise ValueError(
                        f"pair {pair.pair_id}: initial {mapping} differ under frozen codec"
                    )

        self._validated_config = config.model_copy(deep=True)


def prepare_preferences(run, phase, config, catalog, cache):
    key = ("preferences", phase.preferences, phase.validation_preferences)
    if key in cache:
        return cache[key]
    pairs = load_preferences(run.preferences[phase.preferences]).pairs
    validation = (
        load_preferences(run.preferences[phase.validation_preferences]).pairs
        if phase.validation_preferences
        else None
    )
    all_pairs = pairs + (validation or [])
    ids = [pair.pair_id for pair in all_pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("preference pair IDs overlap across manifests")
    names = {span.source for pair in all_pairs for span in (pair.chosen, pair.rejected)}
    if names - run.sources.keys():
        raise ValueError(
            f"preferences reference unknown sources: {sorted(names - run.sources.keys())}"
        )
    corpora = {name: catalog.dataset([name]) for name in sorted(names)}
    data = PreferenceDataset(all_pairs, corpora, config)
    if validation is not None:
        left = {key for pair in pairs for key in data.components(pair)}
        right = {key for pair in validation for key in data.components(pair)}
        if left & right:
            raise ValueError(
                f"preference train/validation components overlap: {next(iter(left & right))}"
            )
    else:
        pairs, validation = split_components(
            pairs, data.components, run.validation_fraction, run.seed
        )
    result = PreferenceData(
        PreferenceDataset(pairs, corpora, config),
        PreferenceDataset(validation, corpora, config),
    )
    cache[key] = result
    return result
