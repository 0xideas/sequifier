"""Offline phase partitioning and frozen-codec validation."""

from dataclasses import dataclass

from sequifier.io.iql_dataset import IQLTransitionDataset
from sequifier.io.policy_dataset import PolicySequenceDataset, split_by_episode
from sequifier.rl.run_state import fingerprint


@dataclass(frozen=True)
class OfflineData:
    training: PolicySequenceDataset
    validation: PolicySequenceDataset

    @property
    def contract(self):
        return [
            {
                "episodes": [
                    fingerprint(episode.model_dump(mode="json"))
                    for episode in part.episodes
                ],
                "episode_ids": part.episode_ids,
                "decisions": len(part),
            }
            for part in (self.training, self.validation)
        ]

    def use_config(self, config):
        for part in (self.training, self.validation):
            part.use_config(config)


def prepare_offline(run, phase, config):
    def corpus(names):
        return PolicySequenceDataset(
            [run.sources[name] for name in names],
            config,
            episode_cache_size=run.episode_cache_size,
        )

    training = corpus(phase.sources)
    if phase.validation_sources:
        validation = corpus(phase.validation_sources)
    else:
        training, validation = split_by_episode(
            training, run.validation_fraction, run.seed, group_identical=True
        )
    if not len(training) or not len(validation):
        raise ValueError(
            f"phases.{phase.name}: nonempty train and validation partitions required"
        )
    for attribute in ("all_episode_ids", "group_ids", "demonstration_keys"):
        if (set(getattr(training, attribute)) & set(getattr(validation, attribute))) - {
            None
        }:
            raise ValueError(
                f"phases.{phase.name}: {attribute} crosses train/validation partitions"
            )
    if phase.algorithm == "iql":
        for part in (training, validation):
            IQLTransitionDataset(part)
    return OfflineData(training, validation)
