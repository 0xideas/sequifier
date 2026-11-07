"""Supervised masked action likelihood using Sequifier's optimizer transaction."""

from dataclasses import dataclass

import torch
from loguru import logger
from torch.utils.data import DataLoader

from sequifier.artifacts.model_artifact import (
    build_model_artifact,
    load_execution_model,
)
from sequifier.artifacts.run_checkpoint import (
    RUN_CHECKPOINT_FORMAT_VERSION,
    DistributedRandomState,
    IntegrationState,
    LoaderState,
    RunCheckpoint,
    RunCheckpointStore,
    select_run_checkpoint,
)
from sequifier.config.policy_train_config import ResolvedPolicyTrainConfig
from sequifier.integration.callbacks import IntegrationManager
from sequifier.integration.contexts import StepIdentity
from sequifier.io.batch import collate_sequence_batches
from sequifier.io.policy_dataset import EpisodeWindowSampler, PolicySequenceDataset
from sequifier.model.factory import InterfaceRuntimeMetadata
from sequifier.objectives import PolicyObjective
from sequifier.rl.policy import CategoricalPolicy
from sequifier.runtime.random_state import RandomState, RandomStateManager
from sequifier.training.loss import LossService, PreparedBatch
from sequifier.training.optimization import OptimizationRuntime, UpdatePolicy
from sequifier.training.runtime import DatasetRuntime, _criterion_modules


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
    run_config: ResolvedPolicyTrainConfig,
    *,
    integrations: IntegrationManager | None = None,
) -> PolicyTrainingResult:
    if not len(training) or not len(validation):
        raise ValueError("policy training and validation partitions must be nonempty")
    if training.config != validation.config:
        raise ValueError(
            "training and validation must share the policy/return definition"
        )
    phase = run_config.training_plan.phases[0]
    config = run_config.optimization
    epochs, seed = phase.epochs, run_config.seed
    if training.config != run_config.policies[phase.policy]:
        raise ValueError("dataset policy must match the resolved run configuration")
    train_ids = tuple(training.episode_ids)
    validation_ids = tuple(validation.episode_ids)
    if set(train_ids) & set(validation_ids):
        raise ValueError(
            "training and validation partitions must not share episode IDs"
        )
    fitted = training.fit_categorical_config()
    training.use_config(fitted)
    validation.use_config(fitted)
    policies = {**run_config.policies, phase.policy: fitted}
    run_config = run_config.model_copy(update={"policies": policies})
    # Seed model initialization and data order independently.
    device = run_config.device
    cuda_index = torch.device(device).index
    cuda_devices = (
        [cuda_index if cuda_index is not None else torch.cuda.current_device()]
        if device.startswith("cuda")
        else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        policy = CategoricalPolicy(training.config, device=device)
        policy.set_normalization_statistics(training.normalization_statistics())
        resume_path = select_run_checkpoint(run_config)
        if run_config.initialize_from is not None and resume_path is None:
            source, source_config, _ = load_execution_model(
                run_config.initialize_from, device=device
            )
            if (
                source_config.model.backbone.architecture
                != policy.execution_config.model.backbone.architecture
            ):
                raise ValueError(
                    "initialize_from requires the same backbone architecture"
                )
            policy.backbone.load_state_dict(source.backbone.state_dict(), strict=True)
            del source
        dataset_config = policy.execution_config.dataset_training[policy.interface_name]
        interface = policy.interface
        dataset_runtime = DatasetRuntime(
            name=policy.interface_name,
            config=dataset_config,
            interface_name=policy.interface_name,
            objective=PolicyObjective(),
            parts={},
            criteria=_criterion_modules(dataset_config, torch.device(device)),
            loss_weights=None,
            class_weights=None,
            frozen_parameter_ids=frozenset(),
            runtime_metadata=InterfaceRuntimeMetadata(
                interface.target_decoder_ids,
                interface.target_n_classes,
                interface.target_global_to_decoder,
            ),
        )
        loss_service = LossService()

        def likelihood(batch):
            batch = batch.to(device)
            features, metadata = policy.model_inputs(batch.model_inputs)
            prepared = PreparedBatch(features, batch.targets, metadata)
            output = policy.network(
                features, metadata, interface_name=policy.interface_name
            )
            return loss_service.calculate(
                output, prepared, dataset_runtime, policy.network
            )

        optimization = OptimizationRuntime.create(
            config, device, policy.network.parameters(), phase_epochs=epochs
        )
        integrations = integrations or IntegrationManager()
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            training,
            batch_size=config.batch_size,
            sampler=EpisodeWindowSampler(training, generator),
            num_workers=config.num_workers,
            generator=generator,
            collate_fn=collate_sequence_batches,
        )
        validation_loader = DataLoader(
            validation,
            batch_size=config.batch_size,
            collate_fn=collate_sequence_batches,
            num_workers=config.num_workers,
        )
        metrics = []
        best_loss = float("inf")
        best_epoch = 0
        best_weights = None
        global_batch = 0
        first_epoch = 0
        random_state = RandomStateManager(torch.device(device))
        store = RunCheckpointStore(run_config, run_config.model_name)
        if resume_path is not None:
            checkpoint = store.load(resume_path)
            previous = checkpoint.training_config
            if not isinstance(previous, ResolvedPolicyTrainConfig):
                raise ValueError("policy resume requires a policy training checkpoint")

            if _resume_contract(previous) != _resume_contract(run_config):
                raise ValueError(
                    "policy resume configuration changes the training contract"
                )
            expected_data = _loader_contract(training, validation)
            if checkpoint.loader_state.parts != expected_data:
                raise ValueError(
                    "policy resume trajectory content or episode order changed"
                )
            policy.network.load_state_dict(
                checkpoint.model.model_state_dict, strict=True
            )
            saved_interface = checkpoint.model.model_config.values["interfaces"][
                policy.interface_name
            ]
            policy.set_normalization_statistics(
                saved_interface["selected_columns_statistics"]
            )
            optimization.optimizer.load_state_dict(checkpoint.optimization.optimizer)
            optimization.load_non_optimizer_state(checkpoint.optimization)
            state = checkpoint.run_state
            first_epoch = state["next_epoch"]
            metrics = list(state["metrics"])
            best_loss, best_epoch, best_weights = (
                state["best_loss"],
                state["best_epoch"],
                state["best_weights"],
            )
            global_batch = state["global_batch"]
            generator.set_state(state["data_generator"])
            random_state.restore(
                RandomState.from_state_dict(checkpoint.random_state.states[0])
            )
            if integrations.enabled:
                integrations.load_state_dict(checkpoint.integration_state.values)
            del checkpoint, state
        optimization.optimizer.zero_grad(set_to_none=True)

        def run_epoch(loader, epoch, *, updating):
            nonlocal global_batch
            policy.train(updating)
            total, decisions = 0.0, 0
            with torch.set_grad_enabled(updating):
                for batch_index, batch in enumerate(loader):
                    result = likelihood(batch)
                    if not torch.isfinite(result.backward_loss):
                        raise ValueError("policy action likelihood became non-finite")
                    total += sum(
                        float(value) for value in result.accounting_sums.values()
                    )
                    decisions += int(result.accounting_count)
                    if updating:
                        identity = StepIdentity(
                            epoch,
                            batch_index,
                            global_batch,
                            optimization.optimizer_step,
                            0,
                            1,
                            0,
                            1,
                        )
                        optimization.accumulate(
                            result.backward_loss, identity, integrations, policy.network
                        )
                        step = optimization.complete_step(
                            policy.network, identity, integrations, UpdatePolicy()
                        )
                        if step.stop_requested or not step.applied:
                            raise RuntimeError(
                                "local policy optimizer did not apply its update"
                            )
                        global_batch += 1
            return total / decisions

        def weights():
            return {
                name: value.detach().cpu().clone()
                for name, value in policy.network.state_dict().items()
            }

        for epoch in range(first_epoch, epochs):
            training_loss = run_epoch(train_loader, epoch, updating=True)
            validation_loss = run_epoch(validation_loader, epoch, updating=False)
            metrics.append(
                {
                    "epoch": float(epoch),
                    "training_nll": training_loss,
                    "validation_nll": validation_loss,
                }
            )
            logger.info(
                f"Policy epoch {epoch + 1}/{epochs}: "
                f"action_nll={training_loss:.6f}, "
                f"validation_action_nll={validation_loss:.6f}"
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_weights = weights()
            optimization.step_scheduler(validation_loss)
            checkpoint = RunCheckpoint(
                format_version=RUN_CHECKPOINT_FORMAT_VERSION,
                model=build_model_artifact(
                    policy.network,
                    policy.execution_config,
                    state_dict=weights(),
                ),
                optimization=optimization.state_dict(),
                run_state={
                    "algorithm": "return_conditioned",
                    "next_epoch": epoch + 1,
                    "metrics": metrics,
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "best_weights": best_weights,
                    "global_batch": global_batch,
                    "data_generator": generator.get_state(),
                },
                random_state=DistributedRandomState(
                    states=(random_state.capture_local().state_dict(),)
                ),
                loader_state=LoaderState(parts=_loader_contract(training, validation)),
                integration_state=IntegrationState(values=integrations.state_dict()),
                training_config=run_config,
            )
            store.save(checkpoint, "latest")
            del checkpoint
        if best_weights is None:
            raise ValueError(
                "policy training did not produce finite validation likelihood"
            )
        policy.network.load_state_dict(best_weights, strict=True)
        policy.eval()
    return PolicyTrainingResult(
        policy,
        tuple(metrics),
        best_epoch,
        optimization.optimizer_step,
        train_ids,
        validation_ids,
    )


def _loader_contract(
    training: PolicySequenceDataset, validation: PolicySequenceDataset
) -> dict:
    return {
        "policy": {
            "training": training.content_keys,
            "validation": validation.content_keys,
            "training_episode_ids": training.episode_ids,
            "validation_episode_ids": validation.episode_ids,
        }
    }


def _resume_contract(config: ResolvedPolicyTrainConfig) -> dict:
    # Device, output paths, workers, and epoch budget may change; learning semantics may not.
    values = config.model_dump(mode="json")
    for key in (
        "device",
        "model_path",
        "report_path",
        "initialize_from",
        "project_root",
    ):
        values.pop(key)
    for phase in values["training_plan"]["phases"]:
        phase.pop("epochs")
    for key in ("resume", "num_workers", "episode_cache_size"):
        values["optimization"].pop(key)
    return values
