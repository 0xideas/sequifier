"""Lazy preference alternatives, connected-component splits and causal scoring."""

import hashlib
import random
from dataclasses import dataclass
from typing import cast

import torch
from torch.nn import functional as F

from sequifier.io.batch import SequifierInputs, collate_model_inputs
from sequifier.io.policy_dataset import PolicySequenceDataset, episode_content_key
from sequifier.io.policy_window import PolicyWindowView, action_index
from sequifier.rl.run_state import fingerprint
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

    def fitting_dataset(self, cache_size):
        refs = {}
        for pair in self.pairs:
            for span in (pair.chosen, pair.rejected):
                index = self.lookup[span.source][span.episode_id]
                refs[(span.source, span.episode_id)] = self.corpora[
                    span.source
                ].episodes.sources[index]
        return PolicySequenceDataset(
            list(refs.values()), self.config, episode_cache_size=cache_size
        )

    def use_config(self, config):
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


def prepare_preferences(run, phase, config):
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
    corpora = {
        name: PolicySequenceDataset(
            [run.sources[name]], config, episode_cache_size=run.episode_cache_size
        )
        for name in names
    }
    data = PreferenceDataset(all_pairs, corpora, config)
    if validation is not None:
        left = {key for pair in pairs for key in data.components(pair)}
        right = {key for pair in validation for key in data.components(pair)}
        if left & right:
            raise ValueError(
                f"preference train/validation components overlap: {next(iter(left & right))}"
            )
    else:
        parents = list(range(len(pairs)))

        def root(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        seen = {}
        for i, pair in enumerate(pairs):
            for key in data.components(pair):
                if key in seen:
                    parents[root(i)] = root(seen[key])
                seen[key] = i
        groups = {}
        for i, pair in enumerate(pairs):
            groups.setdefault(root(i), []).append(pair)
        components = list(groups.values())
        if len(components) < 2:
            raise ValueError(
                "DPO splitting requires at least two independent preference components"
            )
        random.Random(run.seed).shuffle(components)
        count = min(
            len(components) - 1, max(1, int(len(components) * run.validation_fraction))
        )
        validation = [pair for group in components[:count] for pair in group]
        pairs = [pair for group in components[count:] for pair in group]
    return PreferenceData(
        PreferenceDataset(pairs, corpora, config),
        PreferenceDataset(validation, corpora, config),
    )


def sequence_log_probabilities(actor, episode, start, end, chunk_size):
    """One FP32 score per executable decision; context/padding never contribute."""
    actor.eval()
    view = PolicyWindowView(actor.config)
    for offset in range(start, end, chunk_size):
        positions = range(offset, min(offset + chunk_size, end))
        inputs = cast(
            SequifierInputs,
            collate_model_inputs(
                [view.episode_inputs(episode, t, None) for t in positions]
            ),
        ).to(actor.device)
        actions = torch.tensor(
            [action_index(episode.actions[t], actor.config) for t in positions],
            dtype=torch.long,
            device=actor.device,
        )
        scores = actor.evaluate_actions(inputs, actions).float()
        if not torch.isfinite(scores).all():
            raise ValueError("non-finite sequence action log likelihood")
        yield scores


def sequence_score(actor, episode, start, end, chunk_size):
    return torch.stack(
        [
            scores.sum()
            for scores in sequence_log_probabilities(
                actor, episode, start, end, chunk_size
            )
        ]
    ).sum()


def dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta):
    margin = (chosen.float() - rejected.float()) - (
        reference_chosen.float() - reference_rejected.float()
    )
    loss = -F.logsigmoid(beta * margin)
    if not torch.isfinite(margin).all() or not torch.isfinite(loss).all():
        raise ValueError("non-finite DPO margin/loss")
    return loss, margin
