"""Lazy preference storage and leakage-safe partition views."""

import hashlib
from dataclasses import dataclass

import torch

from sequifier.fingerprints import fingerprint
from sequifier.io.episode_corpus import EpisodeView, episode_content_key
from sequifier.io.episode_features import EpisodeFeatures
from sequifier.io.partitioning import split_components
from sequifier.io.policy_window import PolicyWindowView, action_index


@dataclass(frozen=True)
class PreferenceData:
    training: "PreferenceDataset"
    validation: "PreferenceDataset"

    @property
    def contract(self):
        return [self.training.contract, self.validation.contract]

    def fitting_dataset(self):
        return self.training.fitting_dataset()

    def validate(self, interface):
        for partition in (self.training, self.validation):
            partition.validate_features()
            fitting = partition.fitting_dataset()
            fitting.validate_features()
            fitting.validate_normalization(interface)


class PreferenceDataset:
    def __init__(self, pairs, corpora, config):
        self.pairs, self.corpora, self.config = tuple(pairs), corpora, config
        self.lookup = {
            name: {episode: i for i, episode in enumerate(data.episode_ids)}
            for name, data in corpora.items()
        }
        self._fitting = None
        self._features_validated = False
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
        corpus = self.corpora.get(span.source)
        if (
            corpus is not None
            and corpus.exclusions.get(span.episode_id) == "missing_rewards"
        ):
            raise ValueError(
                f"return conditioning requires recorded rewards for episode {span.source}:{span.episode_id}"
            )
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

    def validate_holdout(self):
        for pair in self.pairs:
            for span in (pair.chosen, pair.rejected):
                index = self.lookup[span.source][span.episode_id]
                if self.corpora[span.source].episodes.used_for_fitting(index):
                    raise ValueError(
                        f"pair {pair.pair_id}: indexed metadata was fitted on held-out "
                        f"episode {span.source}:{span.episode_id}"
                    )

    def fitting_dataset(self):
        if self._fitting is None:
            entries = {}
            for pair in self.pairs:
                for span in (pair.chosen, pair.rejected):
                    index = self.lookup[span.source][span.episode_id]
                    entries[(span.source, span.episode_id)] = self.corpora[
                        span.source
                    ].episodes.entries[index]
            self._fitting = EpisodeFeatures(
                EpisodeView(entries.values()),
                self.config,
            )
        return self._fitting

    def validate_features(self):
        if self._features_validated:
            return
        config = self.config
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
            for mapping, lmap, rmap in (
                ("inputs", left.inputs, right.inputs),
                ("metadata", left.metadata, right.metadata),
            ):
                if lmap.keys() != rmap.keys() or any(
                    not torch.equal(lmap[k], rmap[k]) for k in lmap
                ):
                    raise ValueError(
                        f"pair {pair.pair_id}: initial {mapping} differ under frozen codec"
                    )

        self._features_validated = True


def prepare_preferences(run, phase, catalog, cache, membership):
    key = (
        "preferences",
        tuple(ref.source for ref in phase.sources),
        tuple(ref.source for ref in phase.evaluation.sources),
    )
    if key in cache:
        return cache[key]
    config = catalog.config
    corpora = {}

    def read(names):
        pairs = [
            pair
            for name in names
            for pair in catalog.preference_pairs(run.preferences[name])
        ]
        if len({pair.pair_id for pair in pairs}) != len(pairs):
            raise ValueError("Preference pair IDs overlap across manifests")
        for pair in pairs:
            for span in (pair.chosen, pair.rejected):
                if span.source not in corpora:
                    corpora[span.source] = catalog.dataset([span.source])
        return pairs

    def select(ref):
        partition = run.partitions.get(ref.source)
        if partition is None:
            return read([ref.source])
        population = read(partition.sources)
        identity = (
            "pairs",
            tuple(partition.sources),
            partition.validation_fraction,
            run.seed,
        )
        if identity not in membership:
            data = PreferenceDataset(population, corpora, config)
            subsets = split_components(
                population, data.components, partition.validation_fraction, run.seed
            )
            membership[identity] = tuple(
                {pair.pair_id for pair in subset} for subset in subsets
            )
        allowed = membership[identity][int(partition.split == "validation")]
        return [pair for pair in read([partition.part]) if pair.pair_id in allowed]

    pairs = [pair for ref in phase.sources for pair in select(ref)]
    validation = [pair for ref in phase.evaluation.sources for pair in select(ref)]
    all_pairs = pairs + validation
    if (
        not pairs
        or not validation
        or len({pair.pair_id for pair in all_pairs}) != len(all_pairs)
    ):
        raise ValueError("Preference views must be nonempty and have distinct pair IDs")
    data = PreferenceDataset(all_pairs, corpora, config)
    left = {key for pair in pairs for key in data.components(pair)}
    right = {key for pair in validation for key in data.components(pair)}
    if left & right:
        raise ValueError(
            f"Preference train/validation components overlap: {next(iter(left & right))}"
        )
    held_out = PreferenceDataset(validation, corpora, config)
    held_out.validate_holdout()
    result = PreferenceData(
        PreferenceDataset(pairs, corpora, config),
        held_out,
    )
    cache[key] = result
    return result
