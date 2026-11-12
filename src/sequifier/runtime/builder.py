"""Single composition root for complete training runs."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from sequifier.artifacts.backbone_repository import load_revision, select_revision
from sequifier.artifacts.run_checkpoint import RunCheckpointStore, select_run_checkpoint
from sequifier.config.optimizer_config import OptimizerPlan
from sequifier.evaluation.service import EvaluationService
from sequifier.export.service import ExportService
from sequifier.helpers import get_torch_dtype
from sequifier.integration.callbacks import IntegrationManager
from sequifier.model.factory import build_transformer_network, compile_unique_layers
from sequifier.model.parameter_catalog import (
    ParameterCatalog,
    semantic_optimizer_groups,
)
from sequifier.runtime.context import ExecutionEnvironment, RunContext
from sequifier.runtime.random_state import RandomStateManager
from sequifier.training.checkpoint_service import CheckpointRestorer, CheckpointService
from sequifier.training.distributed_strategy import (
    DistributedDataParallelStrategy,
    DistributedStrategy,
    FullyShardedStrategy,
    LocalStrategy,
)
from sequifier.training.loader_state import LoaderStateService
from sequifier.training.loss import LossService
from sequifier.training.metrics_service import MetricsService
from sequifier.training.optimization import GradientPolicy, OptimizationRuntime
from sequifier.training.runtime import (
    DatasetRuntimeRegistry,
    PhaseTrainabilityPlan,
    build_dataset_runtimes,
    configure_phase,
    validate_phase_learning,
)
from sequifier.training.scheduler_budgets import validate_supervised_scheduler_budgets
from sequifier.training.state import RunState


@dataclass
class TrainingRun:
    config: Any
    network: Any
    callable_network: Any
    datasets: DatasetRuntimeRegistry
    optimization: OptimizationRuntime
    optimization_factory: Callable[[int], OptimizationRuntime]
    optimization_phase_index: int
    state: RunState
    distributed: DistributedStrategy
    random: RandomStateManager
    integrations: IntegrationManager
    evaluation: EvaluationService
    exporting: ExportService
    checkpoints: CheckpointService
    metrics: MetricsService
    loss: LossService
    loader_state: LoaderStateService
    context: RunContext

    def start_phase(self, phase_index: int) -> None:
        """Enter the phase with its resolved reset or continuation policy."""

        if self.config.active_phase_index != phase_index:
            configure_phase(
                self.config,
                self.network,
                self.datasets,
                phase_index,
                self.distributed.device,
            )
        optimizer_step = self.optimization.optimizer_step
        if self.config.training_plan[phase_index].optimizer_state == "reset":
            self.optimization = self.optimization_factory(phase_index)
        self.optimization.gradient_policy = GradientPolicy(
            self.config.training_plan[phase_index].accumulation_batches,
            self.config.training_plan[phase_index]
            .optimization["model"]
            .gradient_clip_norm,
        )
        self.optimization.optimizer_step = optimizer_step
        self.optimization_phase_index = phase_index


class RunBuilder:
    def __init__(self, *, semantic_optimizer_grouping: bool = False) -> None:
        self.semantic_optimizer_grouping = semantic_optimizer_grouping

    def _strategy(
        self, config: Any, execution: ExecutionEnvironment
    ) -> DistributedStrategy:
        if not execution.distributed:
            return LocalStrategy(device=execution.device)
        if config.runtime_settings.data_parallelism == "fsdp":
            dtype = None
            if config.runtime_settings.layer_autocast:
                layer_types = config.runtime_settings.layer_type_dtypes or {}
                dtype = get_torch_dtype(layer_types.get("linear", "bfloat16"))
            return FullyShardedStrategy(
                rank=execution.rank,
                local_rank=execution.local_rank,
                world_size=execution.world_size,
                device=execution.device,
                cpu_offload=bool(config.runtime_settings.fsdp_cpu_offload),
                mixed_precision_dtype=dtype,
            )
        return DistributedDataParallelStrategy(
            rank=execution.rank,
            local_rank=execution.local_rank,
            world_size=execution.world_size,
            device=execution.device,
            find_unused_parameters=len(config.dataset_training) > 1,
        )

    def _warm_up(self, run: TrainingRun) -> None:
        from sequifier.model.execution_schema import ExecutionSchema
        from sequifier.model.images import ImageEncoder

        manager = RandomStateManager(run.distributed.device)
        state = manager.capture_local()
        # A single small image reaches ResNet BatchNorm as [1, channels, 1, 1].
        batch_size = (
            2
            if any(
                isinstance(module, ImageEncoder) and not module.frozen
                for module in run.network.modules()
            )
            else 1
        )
        try:
            for dataset in run.datasets.values():
                schema = ExecutionSchema.from_interface(
                    dataset.config.interface,
                    run.config.runtime_settings.context_length,
                )
                features, metadata = schema.bind(
                    schema.example_inputs(batch_size, run.distributed.device)
                )
                schema.validate(features, metadata)
                with torch.no_grad():
                    run.callable_network(
                        features, metadata, interface_name=dataset.interface_name
                    )
        finally:
            manager.restore(state)

    def _construct_loaders(self, run: TrainingRun) -> None:
        """Construct every configured loader before final RNG restoration."""

        for dataset in run.datasets.values():
            for part_name, part in dataset.parts.items():
                part.loader("training")
                if dataset.config.parts[part_name].validation_data_path is not None:
                    part.loader("validation")

    def build(
        self,
        config: Any,
        execution: ExecutionEnvironment,
        integrations: IntegrationManager,
        *,
        initial_artifact=None,
    ) -> TrainingRun:
        strategy = self._strategy(config, execution)
        random_manager = RandomStateManager(execution.device)
        loader_state = LoaderStateService()
        restorer = CheckpointRestorer()
        checkpoint_path = select_run_checkpoint(config)
        loaded = restorer.load(checkpoint_path) if checkpoint_path is not None else None
        if loaded is not None:
            restorer.validate_for_restore(loaded, config)
            state = RunState.from_state_dict(loaded.checkpoint.run_state)
            state.session_id = uuid.uuid4().hex
        else:
            state = RunState()

        validate_phase_learning(config, execution.device)

        from sequifier.export.preflight import ensure_export_preflight

        ensure_export_preflight(config, strategy)
        built = build_transformer_network(
            config,
            device=execution.device,
            initialize=loaded is None,
            logger=logger,
        )
        network = built.network
        if loaded is None and initial_artifact is not None:
            from sequifier.artifacts.model_artifact import read_execution_artifact
            from sequifier.artifacts.model_export import model_execution_config

            artifact = read_execution_artifact(initial_artifact)
            from sequifier.artifacts.model_config import validate_execution_attachment

            validate_execution_attachment(
                artifact.model_config.values, model_execution_config(config)
            )
            from sequifier.artifacts.model_config import load_attached_model_weights

            load_attached_model_weights(network, artifact)
        if loaded is None and initial_artifact is None:
            revision = select_revision(config.model.backbone, config.project_root)
            if revision is not None:
                load_revision(network.backbone, revision)
                state.backbone_parent_revision_id = revision["revision_id"]

        freezing_plan = PhaseTrainabilityPlan.resolve(
            config, network, state.phase_index
        )
        previous_masks = None
        for phase_index, phase in enumerate(config.training_plan):
            masks = PhaseTrainabilityPlan.resolve(
                config, network, phase_index
            ).parameter_ids_by_dataset
            if phase.optimizer_state == "continue" and masks != previous_masks:
                raise ValueError(
                    "optimizer continuation requires identical resolved parameter masks"
                )
            previous_masks = masks

        compile_before_ddp = (
            execution.distributed
            and config.runtime_settings.data_parallelism == "ddp"
            and config.runtime_settings.torch_compile == "outer"
        )
        network_to_prepare = torch.compile(network) if compile_before_ddp else network
        prepared = strategy.prepare_network(network_to_prepare)
        revisions = strategy.gather_objects(state.backbone_parent_revision_id)
        if any(revision != revisions[0] for revision in revisions[1:]):
            raise RuntimeError(
                f"Workers loaded inconsistent backbone revisions: {revisions!r}."
            )
        if loaded is not None:
            restorer.restore_model(loaded, network, strategy)

        datasets = build_dataset_runtimes(
            config,
            network,
            execution.device,
            objectives=built.objectives,
            runtime_metadata=built.runtime_metadata,
            freezing_plan=freezing_plan,
        )
        configure_phase(config, network, datasets, state.phase_index, execution.device)
        for dataset in datasets.values():
            for part in dataset.parts.values():
                part.factory.batch_size = config.training_plan[
                    state.phase_index
                ].batch_size

        parameters: Any = tuple(strategy.prepare_optimizer_parameters(network))
        catalog = ParameterCatalog(network)

        def optimization_factory(phase_index: int) -> OptimizationRuntime:
            phase = config.training_plan[phase_index]
            role = phase.optimization["model"]
            selected_parameters = parameters
            if self.semantic_optimizer_grouping and not isinstance(
                role.optimizer, OptimizerPlan
            ):
                selected_parameters = semantic_optimizer_groups(
                    catalog, parameters={id(parameter) for parameter in parameters}
                )
            return OptimizationRuntime.create(
                role,
                str(execution.device),
                selected_parameters,
                parameter_catalog=catalog,
                accumulation_batches=phase.accumulation_batches,
                layer_type_dtypes=config.runtime_settings.layer_type_dtypes,
            )

        validate_supervised_scheduler_budgets(config, datasets, state.phase_index)
        for phase_index in range(len(config.training_plan)):
            if phase_index != state.phase_index:
                optimization_factory(phase_index)

        optimization = optimization_factory(state.phase_index)
        if loaded is not None:
            restorer.restore_optimization(loaded, optimization, network, strategy)

        callable_network = prepared.callable_network
        if config.runtime_settings.torch_compile == "inner":
            from sequifier.model.encoder_stack import TransformerEncoderStack

            for stack in list(network.modules()):
                if isinstance(stack, TransformerEncoderStack):
                    compile_unique_layers(stack.layers)
        elif (
            config.runtime_settings.torch_compile == "outer" and not compile_before_ddp
        ):
            callable_network = torch.compile(callable_network)

        context = RunContext(
            project_root=Path(config.project_root),
            model_name=config.model_name,
            run_id=state.run_id,
            session_id=state.session_id,
            rank=execution.rank,
            world_size=execution.world_size,
            logger=logger,
        )
        loss_service = LossService()
        run = TrainingRun(
            config=config,
            network=network,
            callable_network=callable_network,
            datasets=datasets,
            optimization=optimization,
            optimization_factory=optimization_factory,
            optimization_phase_index=state.phase_index,
            state=state,
            distributed=strategy,
            random=random_manager,
            integrations=integrations,
            evaluation=EvaluationService(loss_service),
            exporting=ExportService(config, execution.rank),
            checkpoints=CheckpointService(
                RunCheckpointStore(config, config.model_name),
                strategy,
                random_manager,
                loader_state,
            ),
            metrics=MetricsService(context, config),
            loss=loss_service,
            loader_state=loader_state,
            context=context,
        )
        self._construct_loaders(run)
        if loaded is not None:
            restorer.restore_runtime(loaded, run)
            run.state.session_id = context.session_id
        self._warm_up(run)
        if loaded is not None:
            restorer.restore_randomness(loaded, run)
        return run
