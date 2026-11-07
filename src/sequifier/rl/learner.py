"""Algorithm boundary used by trajectory-backed train runs and Python callers."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from sequifier.artifacts.run_checkpoint import checkpoint_path
from sequifier.config.policy_config import PolicyTrainingConfig
from sequifier.config.policy_train_config import (
    ResolvedPolicyTrainConfig,
    TrajectorySourceConfig,
)
from sequifier.io.policy_dataset import TrajectoryDataset, split_by_episode
from sequifier.rl.checkpoint import save_policy
from sequifier.rl.training import PolicyTrainingResult, train_policy
from sequifier.trajectory.importers import source_episodes, source_paths


class AlgorithmRuntime(Protocol):
    def prepare(self) -> None: ...
    def update(self) -> PolicyTrainingResult: ...
    def evaluate(self) -> dict: ...
    def checkpoint(self) -> None: ...


def trajectory_paths(source: TrajectorySourceConfig) -> list[Path]:
    """Compatibility alias for source path/manifest resolution."""
    return source_paths(source)


@dataclass
class ReturnConditionedRuntime:
    config: ResolvedPolicyTrainConfig
    training: TrajectoryDataset = field(init=False)
    validation: TrajectoryDataset = field(init=False)
    excluded_episode_ids: list[str] = field(init=False)
    exclusions: dict[str, str] = field(init=False)
    sources: dict[str, list[str]] = field(init=False)
    result: PolicyTrainingResult = field(init=False)

    def prepare(self) -> None:
        phase = self.config.training_plan.phases[0]
        spec = self.config.policies[phase.policy]
        all_paths: dict[str, list[Path]] = {}
        for name in phase.trajectory_sources + (phase.validation_sources or []):
            all_paths[name] = trajectory_paths(self.config.trajectory_sources[name])
        flattened = [path for paths in all_paths.values() for path in paths]
        protected_paths = set(flattened)
        for name in all_paths:
            manifest = self.config.trajectory_sources[name].manifest
            if manifest is not None:
                protected_paths.add(Path(manifest).resolve())
            source = self.config.trajectory_sources[name]
            for path in (source.metadata_config_path, source.static_data_path):
                if path is not None:
                    protected_paths.add(Path(path).resolve())
        output_paths = [
            Path(self.config.actor_path),
            Path(self.config.report_path),
            checkpoint_path(self.config),
        ]
        if len(output_paths) != len(set(output_paths)):
            raise ValueError("model, report, and run checkpoint paths must differ")
        if set(output_paths) & protected_paths:
            raise ValueError("training outputs must not overwrite trajectory inputs")
        if len(flattened) != len(set(flattened)):
            raise ValueError(
                "trajectory files cannot appear in multiple selected sources"
            )

        def dataset(names: list[str]) -> TrajectoryDataset:
            return TrajectoryDataset(
                [
                    episode
                    for name in names
                    for episode in source_episodes(
                        self.config.trajectory_sources[name], spec
                    )
                ],
                spec,
                episode_cache_size=self.config.optimization.episode_cache_size,
            )

        corpus = dataset(phase.trajectory_sources)
        self.excluded_episode_ids = list(corpus.excluded_episode_ids)
        self.exclusions = dict(corpus.exclusions)
        if phase.validation_sources is None:
            self.training, self.validation = split_by_episode(
                corpus,
                self.config.optimization.validation_fraction,
                self.config.seed,
                group_identical=True,
            )
        else:
            self.training = corpus
            self.validation = dataset(phase.validation_sources)
            self.excluded_episode_ids += self.validation.excluded_episode_ids
            self.exclusions.update(self.validation.exclusions)
            if set(corpus.all_episode_ids) & set(self.validation.all_episode_ids):
                raise ValueError(
                    "duplicate episode IDs across training/validation sources"
                )
        if not len(self.training) or not len(self.validation):
            raise ValueError(
                f"policy training needs nonempty eligible partitions; exclusions: {corpus.exclusions}"
            )
        training_groups = set(self.training.group_ids) - {None}
        validation_groups = set(self.validation.group_ids) - {None}
        if training_groups & validation_groups:
            raise ValueError(
                "collection groups cannot cross training/validation partitions"
            )
        if set(self.training.demonstration_keys) & set(
            self.validation.demonstration_keys
        ):
            raise ValueError(
                "identical trajectories cannot cross training/validation partitions"
            )
        self.sources = {
            name: [str(path) for path in paths] for name, paths in all_paths.items()
        }

    def update(self) -> PolicyTrainingResult:
        phase = self.config.training_plan.phases[0]
        optimization = PolicyTrainingConfig(
            **self.config.optimization.model_dump(),
            epochs=phase.epochs,
            seed=self.config.seed,
        )
        self.result = train_policy(
            self.training, self.validation, optimization, run_config=self.config
        )
        return self.result

    def evaluate(self) -> dict:
        result = self.result
        return {
            "format": "sequifier.policy-training-report",
            "version": 1,
            "algorithm": "return_conditioned",
            "config": self.config.model_dump(mode="json"),
            "sources": self.sources,
            "policy": result.policy.config.model_dump(mode="json"),
            "training_episode_ids": list(result.training_episode_ids),
            "validation_episode_ids": list(result.validation_episode_ids),
            "excluded_episode_ids": self.excluded_episode_ids,
            "exclusions": self.exclusions,
            "training_decisions": len(self.training),
            "validation_decisions": len(self.validation),
            "best_epoch": result.best_epoch,
            "optimizer_steps": result.optimizer_steps,
            "epochs": list(result.metrics),
            "monitor": {
                "metric": "action_nll",
                "direction": "min",
                "value": result.metrics[result.best_epoch]["validation_nll"],
            },
            "artifact_kind": "sequifier_model",
            "actor_path": self.config.actor_path,
        }

    def checkpoint(self) -> None:
        save_policy(
            self.result.policy, self.config.actor_path, metadata=self.evaluate()
        )


def run_policy_training(config: ResolvedPolicyTrainConfig) -> PolicyTrainingResult:
    runtime: AlgorithmRuntime = ReturnConditionedRuntime(config)
    runtime.prepare()
    result = runtime.update()
    report = runtime.evaluate()
    runtime.checkpoint()
    destination = Path(config.report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result
