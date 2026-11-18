"""Adapt resolved phases to native execution and exact recovery."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from sequifier.artifacts.model_artifact import read_execution_artifact
from sequifier.artifacts.phase import PhaseArtifacts
from sequifier.runtime.plan import RunIdentity


@dataclass(frozen=True)
class PortableGroup:
    index: int
    phases: tuple
    incoming: str | None
    resuming: bool
    identity: RunIdentity
    on_checkpoint: Callable


def supervised_group_config(config, phases, index, incoming, resume):
    phase = phases[0]
    from sequifier.artifacts.model_config import (
        extend_prediction_distributions,
        model_execution_from_model_config,
        validate_execution_attachment,
    )
    from sequifier.artifacts.model_export import model_execution_config
    from sequifier.config.components import ResumeConfig
    from sequifier.model.execution import ModelExecution

    training = config.runtime_settings.model_copy(deep=True)
    training.bert_spec = phase.learning.bert_spec
    training.next_occurrence_config = phase.learning.next_occurrence_config
    training.window_stride = phase.learning.window_stride
    if training.distributed:
        raise ValueError("mixed learning plans currently require local phase execution")
    native = PhaseArtifacts.for_phase(
        config.phase_directory, index, phase.name
    ).model.parent
    training.resume = ResumeConfig(
        policy="required" if resume else "never",
        checkpoint_path=str(native / "run.pt"),
    )
    training.log_interval_iterations = 10
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
    if datasets.keys() != names or any(
        not dataset.parts for dataset in datasets.values()
    ):
        raise ValueError(f"{phase.name}: supervised phase requires resolved datasets")

    model = config.model
    interfaces = config.interfaces
    artifact = read_execution_artifact(incoming) if incoming is not None else None
    if artifact is not None:
        source, _ = model_execution_from_model_config(
            artifact.model_config.values, device=config.device
        )
        extend_prediction_distributions(source, config.model)
        model = source.model
        interfaces = source.interfaces
        training.forward_attention = source.runtime_settings.forward_attention or (
            "bidirectional"
            if source.runtime_settings.training_objective == "bert"
            else "causal"
        )
        planned = replace(ModelExecution.from_config(config), runtime_settings=training)
        validate_execution_attachment(
            artifact.model_config.values, model_execution_config(planned)
        )

    resolved = config.model_copy(
        update={
            "model": model,
            "interfaces": interfaces,
            "dataset_training": datasets,
            "training_plan": [item.model_copy(deep=True) for item in phases],
            "runtime_settings": training,
            "seed": config.seed + index,
            "model_name": f"{config.model_name}-phase-{index:03d}-{phase.name}",
            "initialize_from": None,
            "actor_initialization": "attach",
            "curriculum_training": bool(phase.curriculum),
            "curriculum_column": phase.curriculum.column if phase.curriculum else None,
            "export_pt": True,
            "model_path": str(native / "native/model.pt"),
            "report_path": str(native / "native/report.json"),
            "phase_directory": str(native / "native/phases"),
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
        config, request.phases, request.index, request.incoming, request.resuming
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
        identity=request.identity,
        on_checkpoint=request.on_checkpoint,
        backbone_revision=backbone_revision,
    )
    try:
        TrainingEngine().run_group(run)
    finally:
        run.distributed.finalize()


def policy_group_config(config, phases, index, incoming, destination, resume):
    from sequifier.config.training_resolution import policy_run_config

    return policy_run_config(
        config,
        phases,
        incoming=incoming,
        index=index,
        destination=destination,
        resuming=resume,
    )


def run_portable_policy_group(config, request: PortableGroup):
    from sequifier.rl.learner import run_policy_group

    destination = PhaseArtifacts.for_phase(
        config.phase_directory, request.index, request.phases[0].name
    ).model
    run = policy_group_config(
        config,
        request.phases,
        request.index,
        request.incoming,
        destination,
        request.resuming,
    )
    run_policy_group(run, request.identity, request.on_checkpoint)
