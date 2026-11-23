"""Single composition root for complete training runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
from loguru import logger

from sequifier.artifacts.backbone_repository import load_revision, select_revision
from sequifier.artifacts.dependencies import (
    DependencyManifest,
    add_backbone_revision,
    supervised_dependencies,
)
from sequifier.artifacts.phase import CommittedPhases, PhasePublication
from sequifier.artifacts.reports import completed_run_report
from sequifier.artifacts.run_checkpoint import (
    RunCheckpointStore,
    checkpoint_path,
    select_run_checkpoint,
)
from sequifier.config.optimizer_config import OptimizerPlan
from sequifier.evaluation.service import EvaluationService
from sequifier.export.service import ExportService
from sequifier.integration.callbacks import IntegrationManager
from sequifier.model.factory import build_transformer_network, compile_unique_layers
from sequifier.model.parameter_catalog import (
    ParameterCatalog,
    semantic_optimizer_groups,
)
from sequifier.runtime.context import ExecutionEnvironment, RunContext
from sequifier.runtime.plan import RunIdentity
from sequifier.runtime.precision import PrecisionPolicy
from sequifier.runtime.random_state import RandomStateManager
from sequifier.training.checkpoint_service import (
    CheckpointRequest,
    CheckpointRestorer,
    CheckpointService,
)
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
from sequifier.training.plan import CompiledSupervisedPhase, SupervisedRunPlan
from sequifier.training.runtime import (
    DatasetMetrics,
    DatasetRuntimeRegistry,
    PhaseTrainabilityPlan,
    PreparedSupervisedGroup,
    build_dataset_runtimes,
    validate_phase_learning,
)
from sequifier.training.state import RunState


@dataclass
class TrainingRun:
    config: Any
    plan: SupervisedRunPlan
    network: Any
    callable_network: Any
    datasets: DatasetRuntimeRegistry
    optimization: OptimizationRuntime
    dataset_factory: Callable[[CompiledSupervisedPhase], DatasetRuntimeRegistry]
    optimization_factory: Callable[[CompiledSupervisedPhase], OptimizationRuntime]
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
    dependencies: DependencyManifest
    precision: PrecisionPolicy

    def start_phase(
        self,
        phase: CompiledSupervisedPhase,
        datasets: DatasetRuntimeRegistry,
        *,
        continue_optimizer: bool,
    ) -> None:
        """Enter the phase with its resolved reset or continuation policy."""
        optimizer_step = self.optimization.optimizer_step
        if not continue_optimizer:
            self.optimization = self.optimization_factory(phase)
        self.optimization.gradient_policy = GradientPolicy(
            phase.accumulation_batches,
            phase.optimization["model"].gradient_clip_norm,
        )
        self.optimization.optimizer_step = optimizer_step
        self.optimization_phase_index = phase.index
        self.datasets = datasets


class RunBuilder:
    def __init__(self, *, semantic_optimizer_grouping: bool = False) -> None:
        self.semantic_optimizer_grouping = semantic_optimizer_grouping

    def _strategy(
        self, config: Any, execution: ExecutionEnvironment, precision: PrecisionPolicy
    ) -> DistributedStrategy:
        if not execution.distributed:
            return LocalStrategy(device=execution.device)
        if config.run.execution.supervised_settings.distribution.strategy == "fsdp":
            return FullyShardedStrategy(
                rank=execution.rank,
                local_rank=execution.local_rank,
                world_size=execution.world_size,
                device=execution.device,
                cpu_offload=bool(
                    config.run.execution.supervised_settings.distribution.cpu_offload
                ),
                mixed_precision_dtype=precision.autocast_dtype,
            )
        return DistributedDataParallelStrategy(
            rank=execution.rank,
            local_rank=execution.local_rank,
            world_size=execution.world_size,
            device=execution.device,
            find_unused_parameters=max(
                len(config.dataset_training), len(config.interfaces)
            )
            > 1,
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
                    dataset.interface,
                    run.config.execution.options.context_length,
                )
                features, metadata = schema.bind(
                    schema.example_inputs(batch_size, str(run.distributed.device))
                )
                schema.validate(features, metadata)
                with torch.no_grad(), run.precision.autocast():
                    run.callable_network(
                        features, metadata, interface_name=dataset.interface_name
                    )
        finally:
            manager.restore(state)

    def _construct_loaders(self, run: TrainingRun) -> None:
        """Construct active phase loaders before final RNG restoration."""

        phase = run.plan.phases[run.state.phase_index]
        for name, part, role in phase.part_roles():
            run.datasets.resolve(name).parts[part].loader(phase, role)

    def build(
        self,
        config: Any,
        execution: ExecutionEnvironment,
        integrations: IntegrationManager,
        **options,
    ) -> TrainingRun:
        """Construct a runtime even when its plan has already completed."""
        run = self.build_if_needed(
            config, execution, integrations, skip_completed=False, **options
        )
        assert run is not None
        return run

    def build_if_needed(
        self,
        config: Any,
        execution: ExecutionEnvironment,
        integrations: IntegrationManager,
        *,
        skip_completed: bool = True,
        initial_artifact=None,
        phase_offset: int = 0,
        identity: RunIdentity | None = None,
        publication: PhasePublication = PhasePublication(),
        backbone_revision=None,
        dependencies: DependencyManifest | None = None,
    ) -> TrainingRun | None:
        from sequifier.io.output_paths import validate_training_destinations

        group_data = (
            config
            if isinstance(config, PreparedSupervisedGroup)
            else PreparedSupervisedGroup.compile(config)
        )
        config, plan, parts = group_data.config, group_data.plan, group_data.parts
        precision = PrecisionPolicy.from_settings(
            config.run.execution.supervised_settings, execution.device
        )
        strategy = self._strategy(config, execution, precision)
        random_manager = RandomStateManager(execution.device)
        loader_state = LoaderStateService()
        restorer = CheckpointRestorer()
        destination = checkpoint_path(config)
        selected = select_run_checkpoint(destination, config.run.recovery.resume)
        loaded = restorer.load(selected) if selected is not None else None
        if (
            loaded is not None
            and identity is not None
            and loaded.checkpoint.run_state["run_id"] != identity.run_id
        ):
            if config.run.recovery.resume.policy == "required":
                raise ValueError("Supervised recovery belongs to another run")
            loaded = None
        revision = None
        dependencies = dependencies or supervised_dependencies(config, initial_artifact)
        if initial_artifact is None and config.initialize_from is None:
            if loaded is None:
                revision = (
                    select_revision(config.execution.backbone, config.project_root)
                    if backbone_revision is None
                    else backbone_revision or None
                )
            revision_id = (
                loaded.checkpoint.run_state["backbone_parent_revision_id"]
                if loaded is not None
                else revision["revision_id"]
                if revision is not None
                else None
            )
            add_backbone_revision(dependencies, config, revision_id)
        validate_training_destinations(
            config,
            destination,
            dependencies.protected_paths,
            phase_offset=phase_offset,
            group=not publication.standalone,
        )
        validate_phase_learning(config, plan, execution.device)
        if loaded is not None:
            restorer.validate_for_restore(loaded, config, dependencies)
            # Optimizer state uses the parameter order of the saved network.
            interfaces = {
                name: config.interfaces[name]
                for name in loaded.checkpoint.model.model_config.values["interfaces"]
            }
            config.execution = replace(config.execution, interfaces=interfaces)
            state = RunState.from_state_dict(loaded.checkpoint.run_state, plan.phases)
            if skip_completed and publication.standalone and state.phase_commits:
                terminal = (
                    len(state.phase_commits) == len(plan.phases)
                    or state.phase_commits[-1].result["stop_scope"] == "end_plan"
                    or state.phase_commits[-1].result["completion_reason"]
                    == "integration_requested_stop"
                )
                same_outputs = (
                    config.run.export == loaded.checkpoint.training_config.run.export
                    and config.run.outputs
                    == loaded.checkpoint.training_config.run.outputs
                )
                if terminal and same_outputs:
                    done = (
                        strategy.rank == 0
                        and completed_run_report(
                            config.outputs,
                            destination,
                            state.run_id,
                            state.phase_commits,
                        )
                        is not None
                    )
                    if strategy.gather_objects(done)[0]:
                        strategy.finalize()
                        return None
        else:
            state = RunState(
                CommittedPhases(plan.phases, phase_offset), phase_offset=phase_offset
            )
        if state.phase_offset != phase_offset:
            raise ValueError("Supervised recovery has a different phase offset")
        if identity is None:
            identity = strategy.gather_objects(
                RunIdentity.start(state.run_id) if strategy.rank == 0 else None
            )[0]
            if not isinstance(identity, RunIdentity):
                raise RuntimeError("Rank zero did not provide a run identity")
        state.run_id, state.session_id = identity.run_id, identity.session_id

        from sequifier.export.preflight import ensure_export_preflight

        ensure_export_preflight(config, strategy)
        network = build_transformer_network(
            config,
            device=execution.device,
            initialize=loaded is None,
            logger=logger,
        )
        if loaded is None and initial_artifact is not None:
            from sequifier.artifacts.model_artifact import read_execution_artifact
            from sequifier.model.execution import ModelExecution

            artifact = read_execution_artifact(initial_artifact)
            from sequifier.artifacts.model_config import validate_execution_attachment

            validate_execution_attachment(
                artifact.model_config.values, ModelExecution.from_config(config)
            )
            from sequifier.artifacts.model_config import load_attached_model_weights

            load_attached_model_weights(network, artifact)
        if revision is not None:
            load_revision(network.backbone, revision)
            state.backbone_parent_revision_id = revision["revision_id"]

        catalog = ParameterCatalog(network)
        freezing_plans = []
        planned_phases = []
        permanent_masks = []
        for phase in plan.phases:
            freezing_plan = PhaseTrainabilityPlan.resolve(
                config, network, phase, catalog
            )
            masks = freezing_plan.parameter_ids_by_dataset
            if phase.optimizer_state == "continue" and (
                not freezing_plans or freezing_plan != freezing_plans[-1]
            ):
                raise ValueError(
                    "optimizer continuation requires identical resolved parameter masks"
                )
            freezing_plans.append(freezing_plan)
            permanent_masks.extend(set(selected) for selected in masks.values())
            summary = {
                name: sorted(masks[name])
                for name in dict.fromkeys(source.dataset for source in phase.sources)
            }
            planned_phases.append(
                phase.model_copy(update={"trainability_summary": summary})
            )
        plan = SupervisedRunPlan(tuple(planned_phases))
        # Compatibility: avoid building backward graphs for parameters that
        # every dataset in every phase leaves frozen.
        if permanent_masks:
            for parameter_id in set.intersection(*permanent_masks):
                catalog.parameter(parameter_id).requires_grad_(False)

        compile_before_ddp = (
            execution.distributed
            and config.run.execution.supervised_settings.distribution.strategy == "ddp"
            and config.run.execution.supervised_settings.compilation.torch_compile
            == "outer"
        )
        network_to_prepare = (
            cast(torch.nn.Module, torch.compile(network))
            if compile_before_ddp
            else network
        )
        prepared = strategy.prepare_network(network_to_prepare)
        revisions = strategy.gather_objects(state.backbone_parent_revision_id)
        if any(revision != revisions[0] for revision in revisions[1:]):
            raise RuntimeError(
                f"Workers loaded inconsistent backbone revisions: {revisions!r}."
            )
        if loaded is not None:
            restorer.restore_model(loaded, network, strategy)
        dataset_metrics = {name: DatasetMetrics() for name in parts}
        active_phase = plan.phases[state.phase_index]

        if execution.distributed:
            # Distributed preparation can replace parameter objects.
            catalog = ParameterCatalog(network)

        def dataset_factory(view: CompiledSupervisedPhase) -> DatasetRuntimeRegistry:
            return build_dataset_runtimes(
                config,
                view,
                execution.device,
                parts=parts,
                metrics=dataset_metrics,
                masks=freezing_plans[view.index].bind(catalog),
            )

        datasets = dataset_factory(active_phase)
        parameters: Any = tuple(strategy.prepare_optimizer_parameters(network))

        def optimization_factory(phase: CompiledSupervisedPhase) -> OptimizationRuntime:
            role = phase.optimization["model"]
            selected_parameters = parameters
            if self.semantic_optimizer_grouping and not isinstance(
                role.optimizer, OptimizerPlan
            ):
                selected_parameters = semantic_optimizer_groups(
                    catalog, parameters=catalog.parameter_ids(parameters)
                )
            return OptimizationRuntime.create(
                role,
                str(execution.device),
                selected_parameters,
                parameter_catalog=catalog,
                accumulation_batches=phase.accumulation_batches,
                layer_type_dtypes=config.execution.options.layer_type_dtypes,
                autocast_dtype=precision.autocast_dtype,
            )

        optimization = optimization_factory(active_phase)
        if loaded is not None:
            restorer.restore_optimization(loaded, optimization, network, strategy)

        callable_network = prepared.callable_network
        if (
            config.run.execution.supervised_settings.compilation.torch_compile
            == "inner"
        ):
            from sequifier.model.encoder_stack import TransformerEncoderStack

            for stack in list(network.modules()):
                if isinstance(stack, TransformerEncoderStack):
                    compile_unique_layers(stack.layers)
        elif (
            config.run.execution.supervised_settings.compilation.torch_compile
            == "outer"
            and not compile_before_ddp
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
            plan=plan,
            network=network,
            callable_network=callable_network,
            datasets=datasets,
            dataset_factory=dataset_factory,
            optimization=optimization,
            optimization_factory=optimization_factory,
            optimization_phase_index=state.phase_index,
            state=state,
            distributed=strategy,
            random=random_manager,
            integrations=integrations,
            evaluation=EvaluationService(loss_service, precision=precision),
            exporting=ExportService(config, execution.rank),
            checkpoints=CheckpointService(
                RunCheckpointStore(config, config.model_name),
                strategy,
                random_manager,
                loader_state,
                publication=publication,
            ),
            metrics=MetricsService(context, config),
            loss=loss_service,
            loader_state=loader_state,
            context=context,
            dependencies=dependencies,
            precision=precision,
        )
        self._construct_loaders(run)
        if loaded is not None:
            restorer.restore_runtime(loaded, run)
            for name, writer in run.metrics.writers.items():
                writer.restore_positions(
                    state.run_id, state.metric_positions.get(name, {})
                )
        self._warm_up(run)
        if loaded is not None:
            restorer.restore_randomness(loaded, run)
        elif skip_completed:
            run.checkpoints.save(CheckpointRequest("latest", None), run)
        return run
