"""Offline phase partitioning and frozen-codec validation."""

from dataclasses import dataclass, field

from sequifier.io.iql_dataset import IQLTransitionDataset
from sequifier.io.policy_dataset import PolicySequenceDataset, split_by_episode


@dataclass
class OfflineData:
    training: PolicySequenceDataset
    validation: PolicySequenceDataset
    transitions: tuple[IQLTransitionDataset, IQLTransitionDataset] | None = field(
        default=None, init=False
    )

    def prepare_transitions(self):
        if self.transitions is None:
            self.transitions = (
                IQLTransitionDataset(self.training),
                IQLTransitionDataset(self.validation),
            )
        return self.transitions

    @property
    def contract(self):
        return [
            {
                "episodes": part.episode_fingerprints,
                "episode_ids": part.episode_ids,
                "decisions": len(part),
            }
            for part in (self.training, self.validation)
        ]

    def use_config(self, config):
        for part in (self.training, self.validation):
            part.use_config(config)
        if self.transitions is not None:
            for transitions in self.transitions:
                transitions.use_config(config)


def prepare_offline(run, phase, config, catalog):
    corpus = catalog.dataset
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
    return OfflineData(training, validation)
