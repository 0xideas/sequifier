"""Offline phase partitioning and frozen-codec validation."""

from dataclasses import dataclass, field

from sequifier.io.iql_dataset import IQLTransitionDataset
from sequifier.io.policy_dataset import EpisodeDecisions, episode_partition_indices


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

    def fitting_dataset(self):
        return self.training

    def validate(self, interface):
        for part in (self.training, self.validation):
            part.validate_features()
            part.validate_normalization(interface)
        if self.transitions is not None:
            for transitions in self.transitions:
                transitions.validate_normalization(interface)


def prepare_offline(run, phase, catalog, cache, membership):
    # The cache belongs to one run: budgets and optimizers do not affect partitions.
    key = (
        "offline",
        tuple(ref.source for ref in phase.sources),
        tuple(ref.source for ref in (phase.evaluation.sources or [])),
    )
    if key in cache:
        return cache[key]

    def select(refs):
        from sequifier.io.episode_corpus import EpisodeView

        views = []
        for ref in refs:
            partition = run.partitions.get(ref.source)
            if partition is None:
                part = catalog.dataset([ref.source])
            else:
                population = catalog.dataset(partition.sources)
                validate_reward_coverage(population)
                identity = (
                    "episodes",
                    tuple(partition.sources),
                    partition.validation_fraction,
                    run.seed,
                )
                if identity not in membership:
                    indices = episode_partition_indices(
                        population,
                        partition.validation_fraction,
                        run.seed,
                        group_identical=True,
                    )
                    membership[identity] = tuple(
                        {population.episode_ids[i] for i in subset}
                        for subset in indices
                    )
                allowed = membership[identity][int(partition.split == "validation")]
                part = catalog.dataset([partition.part])
                part = part.subset(
                    i
                    for i, episode in enumerate(part.episode_ids)
                    if episode in allowed
                )
            validate_reward_coverage(part)
            views.append(part.episodes)
        return EpisodeDecisions(
            EpisodeView.combine(views),
            catalog.config,
            episode_cache_size=catalog.cache_size,
            cache_owner=catalog.cache_owner,
        )

    training = select(phase.sources)
    validation = select(phase.evaluation.sources)
    for part in (training, validation):
        validate_reward_coverage(part)
    if not len(training) or not len(validation):
        raise ValueError(
            f"phases.{phase.name}: nonempty train and validation partitions required"
        )
    for attribute, train_ids, validation_ids in (
        ("all_episode_ids", training.all_episode_ids, validation.all_episode_ids),
        ("group_ids", training.group_ids, validation.group_ids),
        (
            "demonstration_keys",
            training.demonstration_keys,
            validation.demonstration_keys,
        ),
    ):
        if (set(train_ids) & set(validation_ids)) - {None}:
            raise ValueError(
                f"phases.{phase.name}: {attribute} crosses train/validation partitions"
            )
    data = OfflineData(training, validation)
    cache[key] = data
    return data


def validate_reward_coverage(part):
    if missing := [
        episode
        for episode, reason in part.exclusions.items()
        if reason == "missing_rewards"
    ]:
        raise ValueError(
            f"return conditioning requires recorded rewards for episodes {missing[:10]}"
        )


def prepare_iql(run, phase, catalog, cache, membership):
    decisions = prepare_offline(run, phase, catalog, cache, membership)
    key = ("iql", id(decisions))
    if key not in cache:
        data = OfflineData(decisions.training, decisions.validation)
        data.prepare_transitions()
        cache[key] = data
    return cache[key]
