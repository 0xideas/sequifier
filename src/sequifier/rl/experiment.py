"""Collect, train RC/BC actors, reload, and report achieved control-task returns."""

import json
from itertools import product
from pathlib import Path

import yaml
from loguru import logger

from sequifier.config.policy_config import (
    CategoricalPolicyConfig,
    ControlExperimentConfig,
)
from sequifier.config.rollout_config import AdapterConfig, InputBinding, RolloutConfig
from sequifier.rl.checkpoint import load_policy, save_policy
from sequifier.rl.runtime import collect_policy_rollout
from sequifier.rl.training import train_policy
from sequifier.rollout.control import BinaryControlAdapter
from sequifier.rollout.coordinator import collect_rollout
from sequifier.trajectory.dataset import TrajectoryDataset, split_by_episode
from sequifier.trajectory.returns import return_to_go
from sequifier.trajectory.schema import SeedHistory
from sequifier.trajectory.store import load_trajectory, save_trajectory


def control_seed(horizon: int = 3, context_length: int = 4) -> SeedHistory:
    if horizon < 1 or context_length < 1:
        raise ValueError("control horizon and context_length must be positive")
    return SeedHistory(
        observations={
            "position": [0.0] * context_length,
            "remaining_steps": [0.0] * (context_length - 1) + [float(horizon)],
        },
        attention_valid_mask=[False] * (context_length - 1) + [True],
        static_inputs={},
    )


def control_rollout_config(
    path: str | Path, horizon: int = 3, *, finite_horizon: int | None = None
) -> RolloutConfig:
    return RolloutConfig(
        mode="policy",
        adapter=AdapterConfig(
            factory="sequifier.rollout.control:BinaryControlAdapter",
            config={"horizon": horizon},
        ),
        bindings=[
            InputBinding(input=column, source="environment", key=column)
            for column in ("position", "remaining_steps")
        ],
        seed_index=0,
        trajectory_path=str(path),
        max_episode_steps=finite_horizon,
    )


def collect_control_dataset(
    directory: str | Path,
    *,
    horizon: int = 3,
    repetitions: int = 16,
    context_length: int = 4,
    random_seed: int = 1010,
) -> list[Path]:
    """Enumerate every binary action sequence, repeatedly from identical conditions."""
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or not 1 <= horizon <= 8
    ):
        raise ValueError("dataset control horizon must be an integer in [1, 8]")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
    ):
        raise ValueError("dataset repetitions must be a positive integer")
    destination = Path(directory)
    seed = control_seed(horizon, context_length)
    paths = []
    manifest = []
    for sequence_index, actions in enumerate(product((0.0, 1.0), repeat=horizon)):
        for repetition in range(repetitions):
            path = (
                destination
                / f"sequence-{sequence_index:03d}-episode-{repetition:03d}.json"
            )
            rollout = control_rollout_config(path, horizon)
            action_sequence = iter(actions)

            def predict(_history):
                return {"action": next(action_sequence)}

            trajectory = collect_rollout(
                predict,
                BinaryControlAdapter(horizon),
                seed,
                rollout,
                ["action"],
                horizon,
                random_seed=random_seed,
                metadata={
                    "policy_alignment": "current_observation_previous_action",
                    "behavior": "enumerated_action_sequence",
                    "action_values": [0.0, 1.0],
                    "task": {"name": "binary_control", "horizon": horizon},
                },
            )
            save_trajectory(trajectory, path)
            paths.append(path)
            manifest.append(
                {
                    "path": path.name,
                    "episode_id": trajectory.episode_id,
                    "total_return": trajectory.total_return,
                }
            )
    # Consumers use these exact paths rather than accidentally including stale files.
    (destination / "manifest.json").write_text(
        json.dumps({"version": 1, "episodes": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths


def run_control_experiment(config: ControlExperimentConfig) -> dict:
    root = Path(config.project_root).resolve()
    output = Path(config.output_dir)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    paths = collect_control_dataset(
        output / "dataset",
        horizon=config.horizon,
        repetitions=config.repetitions,
        context_length=config.context_length,
        random_seed=config.training.seed,
    )
    spec = CategoricalPolicyConfig(
        observation_columns=["position", "remaining_steps"],
        context_length=config.context_length,
        backbone=config.backbone,
        returns=config.returns,
    )
    corpus = TrajectoryDataset(paths, spec)
    training, validation = split_by_episode(
        corpus, config.training.validation_fraction, config.training.seed
    )
    (output / "dataset" / "return_labels.json").write_text(
        json.dumps(
            {
                "version": 1,
                "definition": spec.returns.model_dump(mode="json"),
                "episodes": [
                    {"episode_id": episode.episode_id, "return_to_go": labels}
                    for episode, labels in zip(corpus.episodes, corpus.returns)
                ],
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    targets = config.requested_returns
    if targets is None:
        targets = [0.0, return_to_go([1.0] * config.horizon, config.returns.gamma)[0]]
    report = {
        "format": "sequifier.numeric-policy-experiment",
        "version": 1,
        "config": config.model_dump(mode="json"),
        "dataset": {
            "episodes": len(corpus.episodes),
            "decisions": len(corpus),
            "excluded_episode_ids": corpus.excluded_episode_ids,
            "training_episode_ids": [
                episode.episode_id for episode in training.episodes
            ],
            "validation_episode_ids": [
                episode.episode_id for episode in validation.episodes
            ],
            "behavior_mean_return": sum(
                episode.total_return for episode in corpus.episodes
            )
            / len(corpus.episodes),
        },
        "training": {},
        "evaluation": [],
    }
    for algorithm, conditioned in (
        ("return_conditioned", True),
        ("behavior_cloning", False),
    ):
        actor_spec = spec.model_copy(update={"return_conditioned": conditioned})
        train_dataset = TrajectoryDataset(training.episodes, actor_spec)
        validation_dataset = TrajectoryDataset(validation.episodes, actor_spec)
        logger.info(f"Training {algorithm} on {len(train_dataset)} decisions")
        result = train_policy(train_dataset, validation_dataset, config.training)
        policy_path = output / f"{algorithm}.pt"
        save_policy(
            result.policy,
            policy_path,
            metadata={
                "algorithm": algorithm,
                "training_config": config.training.model_dump(mode="json"),
                "training_episode_ids": list(result.training_episode_ids),
                "validation_episode_ids": list(result.validation_episode_ids),
                "best_epoch": result.best_epoch,
            },
        )
        # All reported execution uses weights/transforms reloaded from disk.
        policy = load_policy(policy_path)
        report["training"][algorithm] = {
            "policy_path": str(policy_path),
            "best_epoch": result.best_epoch,
            "optimizer_steps": result.optimizer_steps,
            "epochs": list(result.metrics),
        }
        for target_index, requested_return in enumerate(targets):
            for selection, count in (
                ("deterministic", 1),
                ("sample", config.sampling_episodes),
            ):
                episodes = []
                for episode_index in range(count):
                    path = (
                        output
                        / "evaluation"
                        / f"{algorithm}-return-{target_index}-{selection}-{episode_index}.json"
                    )
                    rollout = control_rollout_config(
                        path, config.horizon, finite_horizon=spec.returns.finite_horizon
                    )
                    trajectory = collect_policy_rollout(
                        policy,
                        BinaryControlAdapter(config.horizon),
                        control_seed(config.horizon, config.context_length),
                        rollout,
                        config.horizon,
                        desired_return=requested_return if conditioned else None,
                        deterministic=selection == "deterministic",
                        random_seed=config.training.seed + episode_index,
                    )
                    save_trajectory(trajectory, path)
                    recorded = load_trajectory(path)
                    achieved = return_to_go(
                        [step.reward for step in recorded.transitions],
                        spec.returns.gamma,
                    )[0]
                    episodes.append(
                        {
                            "episode_id": recorded.episode_id,
                            "trajectory_path": str(path),
                            "actions": [
                                step.predictions[spec.action_column]
                                for step in recorded.transitions
                            ],
                            "achieved_return": achieved,
                            "total_return": recorded.total_return,
                            "remaining_returns": recorded.metadata["policy_execution"][
                                "remaining_returns"
                            ],
                        }
                    )
                report["evaluation"].append(
                    {
                        "algorithm": algorithm,
                        "requested_return": requested_return,
                        "selection": selection,
                        "episodes": episodes,
                        "mean_achieved_return": sum(
                            episode["achieved_return"] for episode in episodes
                        )
                        / count,
                        "mean_absolute_return_error": sum(
                            abs(episode["achieved_return"] - requested_return)
                            for episode in episodes
                        )
                        / count,
                    }
                )
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    logger.info(f"Saved policy learning comparison to '{report_path}'")
    return report


def run_policy_command(config_path: str) -> None:
    with open(config_path, encoding="utf-8") as stream:
        config = ControlExperimentConfig.model_validate(yaml.safe_load(stream))
    run_control_experiment(config)
