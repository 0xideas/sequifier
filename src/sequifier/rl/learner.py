"""Algorithm boundary used by trajectory-backed train runs and Python callers."""

import json
from pathlib import Path

from sequifier.artifacts.run_checkpoint import checkpoint_path
from sequifier.config.policy_train_config import ResolvedPolicyTrainConfig
from sequifier.io.policy_dataset import PolicySequenceDataset, split_by_episode
from sequifier.rl.checkpoint import save_policy
from sequifier.rl.training import PolicyTrainingResult, train_policy
from sequifier.trajectory.importers import source_paths


def run_policy_training(config: ResolvedPolicyTrainConfig) -> PolicyTrainingResult:
    """Prepare episode partitions, train, and publish the selected policy and report."""
    phase = config.training_plan.phases[0]
    spec = config.policies[phase.policy]
    all_paths: dict[str, list[Path]] = {}
    for name in phase.trajectory_sources + (phase.validation_sources or []):
        all_paths[name] = source_paths(config.trajectory_sources[name])
    flattened = [path for paths in all_paths.values() for path in paths]
    protected_paths = set(flattened)
    protected_paths.update(
        Path(path).resolve()
        for name in all_paths
        for source in [config.trajectory_sources[name]]
        for path in (
            source.manifest,
            source.metadata_config_path,
            source.static_data_path,
        )
        if path is not None
    )
    output_paths = [
        Path(config.model_path),
        Path(config.report_path),
        checkpoint_path(config),
    ]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("model, report, and run checkpoint paths must differ")
    if set(output_paths) & protected_paths:
        raise ValueError("training outputs must not overwrite trajectory inputs")
    if len(flattened) != len(set(flattened)):
        raise ValueError("trajectory files cannot appear in multiple selected sources")

    def dataset(names: list[str]) -> PolicySequenceDataset:
        return PolicySequenceDataset(
            [config.trajectory_sources[name] for name in names],
            spec,
            episode_cache_size=config.optimization.episode_cache_size,
        )

    corpus = dataset(phase.trajectory_sources)
    if phase.validation_sources is None:
        training, validation = split_by_episode(
            corpus,
            config.optimization.validation_fraction,
            config.seed,
            group_identical=True,
        )
    else:
        training = corpus
        validation = dataset(phase.validation_sources)
        if set(corpus.all_episode_ids) & set(validation.all_episode_ids):
            raise ValueError("duplicate episode IDs across training/validation sources")
    if not len(training) or not len(validation):
        raise ValueError(
            f"policy training needs nonempty eligible partitions; exclusions: {corpus.exclusions}"
        )
    for attribute in ("group_ids", "demonstration_keys"):
        if (set(getattr(training, attribute)) & set(getattr(validation, attribute))) - {
            None
        }:
            raise ValueError(f"{attribute} cannot cross training/validation partitions")
    exclusions = {**corpus.exclusions, **validation.exclusions}
    sources = {name: [str(path) for path in paths] for name, paths in all_paths.items()}

    result = train_policy(training, validation, config)
    report = {
        "format": "sequifier.policy-training-report",
        "version": 1,
        "algorithm": "return_conditioned",
        "config": config.model_dump(mode="json"),
        "sources": sources,
        "policy": result.policy.config.model_dump(mode="json"),
        "training_episode_ids": list(result.training_episode_ids),
        "validation_episode_ids": list(result.validation_episode_ids),
        "excluded_episode_ids": list(exclusions),
        "exclusions": exclusions,
        "training_decisions": len(training),
        "validation_decisions": len(validation),
        "best_epoch": result.best_epoch,
        "optimizer_steps": result.optimizer_steps,
        "epochs": list(result.metrics),
        "monitor": {
            "metric": "action_nll",
            "direction": "min",
            "value": result.metrics[result.best_epoch]["validation_nll"],
        },
        "artifact_kind": "sequifier_model",
        "model_path": config.model_path,
    }
    save_policy(result.policy, config.model_path, metadata=report)
    destination = Path(config.report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result
