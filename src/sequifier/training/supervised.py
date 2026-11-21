"""Launch supervised workers and execute a native supervised run."""

import os
from typing import Any

import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn

from sequifier.artifacts.dependencies import supervised_dependencies
from sequifier.artifacts.run_checkpoint import checkpoint_path
from sequifier.config.train_config import ResolvedSequifierConfig as TrainModel
from sequifier.distributed.env import setup_distributed_env
from sequifier.helpers import configure_determinism, configure_logger
from sequifier.integration import IntegrationManager, IntegrationSpec
from sequifier.io.output_paths import validate_training_destinations
from sequifier.runtime.builder import RunBuilder
from sequifier.runtime.context import ExecutionEnvironment
from sequifier.training.engine import TrainingEngine
from sequifier.training.runtime import PreparedSupervisedGroup
from sequifier.typechecking import beartype


def execute_supervised(
    group,
    execution,
    *,
    integration_specs=(),
    integration_instances=(),
    semantic_optimizer_grouping=False,
    **build_options,
):
    """Build and finalize one native runtime in the supplied worker environment."""
    if isinstance(group, PreparedSupervisedGroup):
        config, dataset_names = group.config, ()
    else:
        config, dataset_names = group, tuple(group.dataset_training)
    settings = config.run.execution.supervised_settings
    torch.set_float32_matmul_precision(settings.precision.float32_matmul_precision)
    logger = configure_logger(
        config.project_root,
        config.model_name,
        execution.rank,
        dataset_names=dataset_names,
        rank_specific=execution.distributed,
    )
    configure_determinism(config.seed, config.run.reproducibility.enforce_determinism)
    integrations = IntegrationManager(
        specs=integration_specs,
        instances=integration_instances,
        rank=execution.rank,
        world_size=execution.world_size,
        distributed=execution.distributed,
    )
    integrations.validate_execution(
        torch_compile=settings.compilation.torch_compile,
        data_parallelism=settings.distribution.strategy,
    )
    run = RunBuilder(semantic_optimizer_grouping=semantic_optimizer_grouping).build(
        group, execution, integrations, **build_options
    )
    try:
        logger.info(
            f"Starting training for {run.context.model_name}, run {run.state.run_id}"
        )
        result = TrainingEngine().run(run)
        logger.info(f"Training complete ({result.completion_reason})")
    finally:
        run.distributed.finalize()


def _execution_device(config: TrainModel, local_rank: int) -> torch.device:
    configured = torch.device(config.device)
    if configured.type == "cuda":
        return torch.device("cuda", local_rank)
    return configured


@beartype
def train_worker(
    local_rank: int,
    world_size: int,
    config: TrainModel,
    integration_specs: tuple[IntegrationSpec, ...] = (),
    integration_instances: tuple[Any, ...] = (),
    semantic_optimizer_grouping: bool = False,
    *,
    global_rank: int | None = None,
) -> None:
    """Initialize one execution environment and run the composed runtime."""

    global_rank = local_rank if global_rank is None else global_rank
    distribution = config.run.execution.supervised_settings.distribution
    dependencies = supervised_dependencies(config)
    validate_training_destinations(
        config, checkpoint_path(config), dependencies.protected_paths
    )
    execution = ExecutionEnvironment(
        rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        device=_execution_device(config, local_rank),
        distributed=distribution.distributed,
    )
    if distribution.strategy != "local":
        if execution.device.type == "cuda":
            torch.cuda.set_device(local_rank)
        setup_distributed_env(global_rank, local_rank, world_size, distribution.backend)
    try:
        execute_supervised(
            config,
            execution,
            integration_specs=integration_specs,
            integration_instances=integration_instances,
            semantic_optimizer_grouping=semantic_optimizer_grouping,
            initial_artifact=config.initialize_from,
            dependencies=dependencies,
        )
    finally:
        if execution.distributed and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def launch_supervised(
    config: TrainModel,
    *,
    integration_specs: tuple[IntegrationSpec, ...],
    integration_instances: tuple[Any, ...],
    semantic_optimizer_grouping: bool,
) -> None:
    """Launch the native supervised runtime in its execution environment."""
    distribution = config.run.execution.supervised_settings.distribution
    if distribution.distributed and integration_instances:
        raise ValueError(
            "Distributed runs require IntegrationSpec; direct instances cannot "
            "be transferred to workers."
        )
    if distribution.distributed and not {"RANK", "WORLD_SIZE"} <= os.environ.keys():
        spawn(
            train_worker,
            args=(
                distribution.world_size,
                config,
                integration_specs,
                (),
                semantic_optimizer_grouping,
            ),
            nprocs=distribution.world_size,
            join=True,
        )
        return
    train_worker(
        int(os.environ.get("LOCAL_RANK", 0)) if distribution.distributed else 0,
        int(os.environ["WORLD_SIZE"]) if distribution.distributed else 1,
        config,
        integration_specs,
        integration_instances,
        semantic_optimizer_grouping,
        global_rank=int(os.environ["RANK"]) if distribution.distributed else 0,
    )
