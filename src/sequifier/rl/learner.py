"""Algorithm boundary used by trajectory-backed train runs and Python callers."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from sequifier.artifacts.run_checkpoint import checkpoint_path
from sequifier.config.policy_config import PolicyTrainingConfig
from sequifier.config.policy_train_config import (
    ResolvedPolicyTrainConfig,
    TrajectorySourceConfig,
)
from sequifier.rl.checkpoint import save_policy
from sequifier.rl.training import PolicyTrainingResult, train_policy
from sequifier.trajectory.dataset import TrajectoryDataset, split_by_episode
from sequifier.trajectory.store import load_trajectory


class AlgorithmRuntime(Protocol):
    def prepare(self) -> None: ...
    def update(self) -> PolicyTrainingResult: ...
    def evaluate(self) -> dict: ...
    def checkpoint(self) -> None: ...


def trajectory_paths(source: TrajectorySourceConfig) -> list[Path]:
    if source.manifest is None:
        paths = [Path(path) for path in source.paths]
    else:
        manifest_path = Path(source.manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("version") != 1:
            raise ValueError("trajectory manifest must have version 1")
        episodes = manifest.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError("trajectory manifest requires nonempty episodes")
        paths = []
        for entry in episodes:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("path"), str)
                or not entry["path"]
            ):
                raise ValueError("each manifest episode requires a nonempty path")
            path = manifest_path.parent / entry["path"]
            if "episode_id" in entry:
                if load_trajectory(path).episode_id != entry["episode_id"]:
                    raise ValueError(f"manifest episode ID disagrees with {path}")
            paths.append(path)
    paths = [path.resolve() for path in paths]
    if len(paths) != len(set(paths)):
        raise ValueError("trajectory source contains duplicate paths")
    return paths


@dataclass
class ReturnConditionedRuntime:
    config: ResolvedPolicyTrainConfig
    training: TrajectoryDataset = field(init=False)
    validation: TrajectoryDataset = field(init=False)
    excluded_episode_ids: list[str] = field(init=False)
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

        # Validate identities incrementally, including episodes without exact returns.
        ids = [load_trajectory(path).episode_id for path in flattened]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate episode IDs across trajectory sources")

        def dataset(names: list[str]) -> TrajectoryDataset:
            semantics: set[Literal["policy", "executed_predictions"]] = {
                self.config.trajectory_sources[name].action_semantics for name in names
            }
            if len(semantics) != 1:
                raise ValueError("selected sources must share action_semantics")
            return TrajectoryDataset(
                [path for name in names for path in all_paths[name]],
                spec,
                action_semantics=next(iter(semantics)),
                episode_cache_size=self.config.optimization.episode_cache_size,
            )

        corpus = dataset(phase.trajectory_sources)
        self.excluded_episode_ids = list(corpus.excluded_episode_ids)
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
        if set(self.training.content_keys) & set(self.validation.content_keys):
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
            "training_episode_ids": list(result.training_episode_ids),
            "validation_episode_ids": list(result.validation_episode_ids),
            "excluded_episode_ids": self.excluded_episode_ids,
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
