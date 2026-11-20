"""Adapt resolved phases to native execution and exact recovery."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from sequifier.artifacts.phase import PhaseArtifacts, PhaseCommit
from sequifier.runtime.plan import RunIdentity


@dataclass(frozen=True)
class PortableGroup:
    index: int
    phases: tuple
    incoming: str | None
    resuming: bool
    identity: RunIdentity
    on_checkpoint: Callable[[Path, PhaseCommit | None], None]


def supervised_group_config(config, phases, index, resume):
    phase = phases[0]
    from sequifier.config.components import ResumeConfig

    artifacts = PhaseArtifacts.for_phase(config.phase_directory, index, phase.name)
    resume = ResumeConfig(
        policy="required" if resume else "never",
        checkpoint_path=str(artifacts.recovery),
    )
    names = {
        ref.dataset
        for item in phases
        for ref in [*item.sources, *item.evaluation_sources]
    }
    datasets = {
        name: dataset
        for name, dataset in config.dataset_training.items()
        if name in names
    }
    resolved = config.model_copy(
        update={
            "dataset_training": datasets,
            "training_plan": list(phases),
            "runtime_settings": config.runtime_settings.model_copy(
                update={"resume": resume}
            ),
            "execution": replace(
                config.execution,
                seed=phase.resolved_seed
                if phase.resolved_seed is not None
                else config.seed + index,
                model_name=f"{config.model_name}-phase-{index:03d}-{phase.name}",
            ),
            "initialize_from": None,
            "actor_initialization": "attach",
            "curriculum_training": bool(phase.curriculum),
            "curriculum_column": phase.curriculum.column if phase.curriculum else None,
        }
    )
    return resolved


def run_supervised_group(config, request: PortableGroup, backbone_revision):
    from sequifier.helpers import configure_determinism, configure_logger
    from sequifier.integration import IntegrationManager
    from sequifier.runtime.builder import RunBuilder
    from sequifier.runtime.context import ExecutionEnvironment
    from sequifier.training.engine import TrainingEngine

    resolved = supervised_group_config(
        config, request.phases, request.index, request.resuming
    )
    torch.set_float32_matmul_precision(
        resolved.runtime_settings.float32_matmul_precision
    )
    configure_logger(config.project_root, resolved.model_name, 0)
    configure_determinism(resolved.seed, resolved.runtime_settings.enforce_determinism)
    environment = ExecutionEnvironment(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device(config.device),
        distributed=False,
    )
    integrations = IntegrationManager(rank=0, world_size=1, distributed=False)
    run = RunBuilder().build(
        resolved,
        environment,
        integrations,
        initial_artifact=request.incoming,
        phase_offset=request.index,
        identity=request.identity,
        on_checkpoint=request.on_checkpoint,
        group=True,
        backbone_revision=backbone_revision,
    )
    try:
        TrainingEngine().run_group(run)
    finally:
        run.distributed.finalize()


def policy_group_config(config, phases, index, incoming=None, resuming=False):
    from sequifier.config.training_resolution import policy_run_config

    return policy_run_config(
        config,
        phases,
        incoming=incoming,
        index=index,
        resuming=resuming,
    )


def run_portable_policy_group(config, request: PortableGroup, prepared, preparation):
    from sequifier.rl.learner import run_policy_group

    run = policy_group_config(
        config,
        request.phases,
        request.index,
        request.incoming,
        request.resuming,
    )
    run_policy_group(
        run,
        request.identity,
        request.on_checkpoint,
        prepared=prepared,
        preparation=preparation,
    )
