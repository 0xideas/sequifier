"""Single composition root for complete training runs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from sequifier.artifacts.backbone_repository import load_revision, select_revision
from sequifier.artifacts.run_checkpoint import RunCheckpointStore, select_run_checkpoint
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
from sequifier.training.optimization import OptimizationRuntime
from sequifier.training.runtime import DatasetRuntimeRegistry, build_dataset_runtimes
from sequifier.training.state import RunState


@dataclass
class TrainingRun:
    config: Any
    network: Any
    callable_network: Any
    datasets: DatasetRuntimeRegistry
    optimization: OptimizationRuntime
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


class RunBuilder:
    def __init__(self, *, semantic_optimizer_grouping: bool = False) -> None:
        self.semantic_optimizer_grouping = semantic_optimizer_grouping

    def _strategy(
        self, config: Any, execution: ExecutionEnvironment
    ) -> DistributedStrategy:
        if not execution.distributed:
            return LocalStrategy(device=execution.device)
        if config.global_training.data_parallelism == "fsdp":
            dtype = None
            if config.global_training.layer_autocast:
                layer_types = config.global_training.layer_type_dtypes or {}
                dtype = get_torch_dtype(layer_types.get("linear", "bfloat16"))
            return FullyShardedStrategy(
                rank=execution.rank,
                local_rank=execution.local_rank,
                world_size=execution.world_size,
                device=execution.device,
                cpu_offload=bool(config.global_training.fsdp_cpu_offload),
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
        dataset = next(iter(run.datasets.values()))
        interface = dataset.config.interface
        batch_size = 1
        features = {
            column: torch.ones(
                (batch_size, run.config.global_training.context_length),
                dtype=(
                    torch.int64
                    if column in interface.categorical_columns
                    else torch.float32
                ),
                device=run.distributed.device,
            )
            for column in interface.input_columns
        }
        metadata = {
            "attention_valid_mask": torch.ones(
                (batch_size, run.config.global_training.context_length),
                dtype=torch.bool,
                device=run.distributed.device,
            )
        }
        with torch.no_grad():
            run.callable_network(
                features, metadata, interface_name=dataset.interface_name
            )

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

        built = build_transformer_network(
            config,
            device=execution.device,
            initialize=loaded is None,
            logger=logger,
        )
        network = built.network
        if loaded is None:
            revision = select_revision(config.model.backbone, config.project_root)
            if revision is not None:
                load_revision(network.backbone, revision)
                state.backbone_parent_revision_id = revision["revision_id"]

        compile_before_ddp = (
            execution.distributed
            and config.global_training.data_parallelism == "ddp"
            and config.global_training.torch_compile == "outer"
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
        )
        parameters: Any = tuple(strategy.prepare_optimizer_parameters(network))
        if self.semantic_optimizer_grouping:
            parameters = semantic_optimizer_groups(
                ParameterCatalog(network),
                parameters={id(parameter) for parameter in parameters},
            )
        optimization = OptimizationRuntime.create(
            config.global_training, str(execution.device), parameters
        )
        if loaded is not None:
            restorer.restore_optimization(loaded, optimization, network, strategy)

        callable_network = prepared.callable_network
        if config.global_training.torch_compile == "inner":
            compile_unique_layers(network.backbone.layers)
        elif config.global_training.torch_compile == "outer" and not compile_before_ddp:
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
