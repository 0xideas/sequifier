"""Offline phase partitioning and frozen-codec validation."""

from dataclasses import dataclass, field

from sequifier.io.iql_dataset import IQLTransitionDataset
from sequifier.io.policy_dataset import EpisodeDecisions, split_by_episode


@dataclass
class OfflineData:
    training: EpisodeDecisions
    validation: EpisodeDecisions
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

    def fitting_dataset(self):
        return self.training

    def validate(self, actor):
        self.use_config(actor.config)
        for part in (self.training, self.validation):
            part.validate_normalization(actor.interface.selected_columns_statistics)
        if self.transitions is not None:
            for transitions in self.transitions:
                transitions.validate_normalization(actor)


def prepare_offline(run, phase, config, catalog, cache):
    # The cache belongs to one run: budgets and optimizers do not affect partitions.
    key = ("offline", tuple(phase.sources), tuple(phase.validation_sources or []))
    if key in cache:
        return cache[key]
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
    data = OfflineData(training, validation)
    cache[key] = data
    return data


def prepare_iql(run, phase, config, catalog, cache):
    data = prepare_offline(run, phase, config, catalog, cache)
    data.prepare_transitions()
    return data
