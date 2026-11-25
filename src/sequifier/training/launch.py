"""Own process membership, device binding and the run lock across all phases."""

import os
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn

from sequifier.artifacts.dependencies import supervised_dependencies
from sequifier.artifacts.run_checkpoint import checkpoint_path, training_lock
from sequifier.config.train_config import ResolvedSequifierConfig as TrainModel
from sequifier.config.training_resolution import prepare_locked_run
from sequifier.distributed.env import setup_distributed_env
from sequifier.integration import IntegrationSpec
from sequifier.io.output_paths import validate_training_destinations
from sequifier.runtime.context import ExecutionEnvironment
from sequifier.typechecking import beartype


def _execution_device(config: TrainModel, local_rank: int) -> torch.device:
    configured = torch.device(config.device)
    if (
        configured.type == "cuda"
        and config.run.execution.distribution_settings.distributed
    ):
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
):
    """Initialize one execution environment and run the composed runtime."""

    global_rank = local_rank if global_rank is None else global_rank
    distribution = config.run.execution.distribution_settings
    execution = ExecutionEnvironment(
        rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        device=_execution_device(config, local_rank),
        distributed=distribution.distributed,
    )
    owns_group = execution.distributed and not dist.is_initialized()
    if distribution.strategy != "local":
        if execution.device.type == "cuda":
            torch.cuda.set_device(local_rank)
        setup_distributed_env(global_rank, local_rank, world_size, distribution.backend)
    try:
        if distribution.strategy != "local" and distribution.backend == "nccl":
            # Object collectives carry CPU simulator buffers and recovery snapshots.
            execution = replace(execution, control_group=dist.new_group(backend="gloo"))
        lock = (
            training_lock(checkpoint_path(config))
            if global_rank == 0
            else nullcontext()
        )
        with lock:
            if global_rank == 0:
                config = prepare_locked_run(config)
            if execution.distributed:
                seed = [config.seed]
                dist.broadcast_object_list(seed, src=0, group=execution.control_group)
                if global_rank != 0:
                    config = prepare_locked_run(config, seed[0])
            if config.run_mode != "supervised":
                from sequifier.training.coordinator import execute_policy_plan

                return execute_policy_plan(config, execution)
            from sequifier.training.supervised import execute_supervised

            dependencies = supervised_dependencies(config)
            validate_training_destinations(
                config, checkpoint_path(config), dependencies.protected_paths
            )
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
        if execution.control_group is not None:
            dist.destroy_process_group(execution.control_group)
        if owns_group and dist.is_initialized():
            dist.destroy_process_group()


def launch_training(
    config: TrainModel,
    *,
    integration_specs: tuple[IntegrationSpec, ...],
    integration_instances: tuple[Any, ...],
    semantic_optimizer_grouping: bool,
):
    """Launch the plan under torchrun, local spawn, or an in-process worker."""
    distribution = config.run.execution.distribution_settings
    if distribution.distributed and integration_instances:
        raise ValueError(
            "Distributed runs require IntegrationSpec; direct instances cannot "
            "be transferred to workers."
        )
    if distribution.strategy != "local" and dist.is_initialized():
        if dist.get_backend() != distribution.backend:
            raise ValueError("The initialized process group uses a different backend")
        local_rank = int(
            os.environ.get(
                "LOCAL_RANK",
                torch.cuda.current_device()
                if torch.device(config.device).type == "cuda"
                else 0,
            )
        )
        return train_worker(
            local_rank,
            dist.get_world_size(),
            config,
            integration_specs,
            (),
            semantic_optimizer_grouping,
            global_rank=dist.get_rank(),
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
        if config.run_mode != "supervised":
            from sequifier.training.coordinator import published_training_result

            return published_training_result(config)
        return None
    return train_worker(
        int(os.environ.get("LOCAL_RANK", 0)) if distribution.distributed else 0,
        int(os.environ["WORLD_SIZE"]) if distribution.distributed else 1,
        config,
        integration_specs,
        integration_instances,
        semantic_optimizer_grouping,
        global_rank=int(os.environ["RANK"]) if distribution.distributed else 0,
    )
