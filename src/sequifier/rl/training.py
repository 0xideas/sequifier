"""Policy datasets adapted to Sequifier's ordinary TrainingRun and TrainingEngine."""

from dataclasses import dataclass

import torch

from sequifier.artifacts.model_artifact import load_execution_model
from sequifier.config.train_config import (
    EvaluationMonitorSpecModel,
    PolicyTrainConfig,
    ResolvedTrainingPhase,
    ResolvedTrainingSource,
)
from sequifier.integration.callbacks import IntegrationManager
from sequifier.io.policy_dataset import PolicySequenceDataset
from sequifier.rl.policy import CategoricalPolicy, policy_execution_config
from sequifier.runtime.builder import RunBuilder, TrainingRun
from sequifier.runtime.context import ExecutionEnvironment
from sequifier.training.engine import TrainingEngine
from sequifier.training.runtime import DatasetPartRuntime, PartLoaderFactory


@dataclass(frozen=True)
class PolicyTrainingResult:
    policy: CategoricalPolicy
    metrics: tuple[dict[str, float], ...]
    best_epoch: int
    optimizer_steps: int
    training_episode_ids: tuple[str, ...]
    validation_episode_ids: tuple[str, ...]


def train_policy(
    training: PolicySequenceDataset,
    validation: PolicySequenceDataset,
    run_config: PolicyTrainConfig,
    *,
    integrations: IntegrationManager | None = None,
) -> PolicyTrainingResult:
    phase = run_config.training_plan.phases[0]
    if phase.algorithm != "policy_imitation":
        raise ValueError("train_policy requires a policy_imitation phase")
    if not len(training) or not len(validation):
        raise ValueError("policy training and validation partitions must be nonempty")
    if training.config != validation.config or training.config != run_config.policy:
        raise ValueError(
            "dataset policies must match the resolved training configuration"
        )
    if set(training.episode_ids) & set(validation.episode_ids):
        raise ValueError(
            "training and validation partitions must not share episode IDs"
        )
    fitted = training.fit_categorical_config()
    for dataset in (training, validation):
        dataset.use_config(fitted)
    name, part_name = fitted.interface_name, "episodes"
    source = ResolvedTrainingSource(
        source=f"{name}.{part_name}", dataset=name, part=part_name
    )
    config = policy_execution_config(
        fitted,
        run_config.model,
        run_config.device,
        global_training=run_config.global_training,
        statistics=training.normalization_statistics(),
        project_root=run_config.project_root,
        model_name=run_config.model_name,
        seed=run_config.seed,
        training_plan=[
            ResolvedTrainingPhase(
                name=phase.name,
                algorithm="policy_imitation",
                epochs=phase.epochs,
                mode="sequential",
                sources=[source],
            )
        ],
        evaluation_sources=[source],
        evaluation_monitor=EvaluationMonitorSpecModel(
            source=source.source, metric="loss", mode="min"
        ),
    )

    def configure(run: TrainingRun, resuming: bool) -> None:
        run.datasets.resolve(name).parts[part_name] = DatasetPartRuntime(
            part_name,
            PartLoaderFactory(
                config,
                name,
                part_name,
                {"training": training, "validation": validation},
            ),
        )
        if not resuming and run_config.initialize_from is not None:
            source_network, source_config, _ = load_execution_model(
                run_config.initialize_from, device=config.device
            )
            if (
                source_config.model.backbone.architecture
                != config.model.backbone.architecture
            ):
                raise ValueError(
                    "initialize_from requires the same backbone architecture"
                )
            run.network.backbone.load_state_dict(
                source_network.backbone.state_dict(), strict=True
            )

    device = torch.device(config.device)
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(config.seed)
        run = RunBuilder().build(
            config,
            ExecutionEnvironment(0, 0, 1, device, False),
            integrations or IntegrationManager(),
            configure=configure,
        )
        try:
            TrainingEngine().run(run)
            if run.state.best_model_state_dict is None:
                raise ValueError(
                    "policy training did not produce finite validation likelihood"
                )
            run.network.load_state_dict(run.state.best_model_state_dict, strict=True)
            policy = CategoricalPolicy.from_network(run.network, config, name)
            policy.eval()
            metrics = tuple(
                {
                    "epoch": float(item["epoch"] - 1),
                    "training_nll": item["training"][name],
                    "validation_nll": item["validation"][source.source],
                }
                for item in run.state.epoch_metrics
            )
            return PolicyTrainingResult(
                policy,
                metrics,
                run.state.best_epoch - 1,
                run.optimization.optimizer_step,
                tuple(training.episode_ids),
                tuple(validation.episode_ids),
            )
        finally:
            run.distributed.finalize()
