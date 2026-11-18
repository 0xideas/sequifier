"""Adapt resolved phases to native execution and exact recovery."""

from dataclasses import dataclass, replace

import torch

from sequifier.artifacts.model_artifact import ModelArtifact, read_execution_artifact
from sequifier.artifacts.phase import PhaseArtifacts
from sequifier.runtime.plan import RunIdentity


@dataclass(frozen=True)
class PortablePhaseRequest:
    index: int
    incoming: str | None
    artifacts: PhaseArtifacts
    resuming: bool
    legacy_recovery: bool
    identity: RunIdentity


@dataclass(frozen=True)
class PortablePhaseResult:
    artifact: ModelArtifact
    report: dict


def supervised_phase_config(config, phase, index, incoming, resume):
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
        policy="if_exists" if resume else "never",
        checkpoint_path=str(native / "run.pt"),
    )
    training.log_interval_iterations = 10
    names = {ref.dataset for ref in [*phase.sources, *phase.evaluation_sources]}
    datasets = {
        name: dataset.model_copy(update=phase.learning.datasets[name].model_dump())
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
            "training_plan": [phase.model_copy(deep=True)],
            "active_phase_index": 0,
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


def run_supervised_phase(
    config, phase, request: PortablePhaseRequest
) -> PortablePhaseResult:
    from sequifier.helpers import configure_determinism, configure_logger
    from sequifier.integration import IntegrationManager
    from sequifier.runtime.builder import RunBuilder
    from sequifier.runtime.context import ExecutionEnvironment
    from sequifier.training.engine import TrainingEngine

    resolved = supervised_phase_config(
        config, phase, request.index, request.incoming, request.resuming
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
        legacy_recovery=request.legacy_recovery,
    )
    try:
        report = TrainingEngine().run_phase(run)
        return PortablePhaseResult(
            read_execution_artifact(report["model_path"]),
            {
                **report,
                "resolved_seed": resolved.seed,
                "optimizer_steps": run.state.optimizer_step,
            },
        )
    finally:
        run.distributed.finalize()


def policy_phase_config(config, phase, index, incoming, destination, resume):
    from sequifier.config.training_resolution import policy_run_config

    return policy_run_config(
        config,
        [phase],
        incoming=incoming,
        index=index,
        destination=destination,
        resuming=resume,
    )


def run_portable_policy_phase(
    config, phase, request: PortablePhaseRequest
) -> PortablePhaseResult:
    from sequifier.rl.learner import run_policy_phase

    run = policy_phase_config(
        config,
        phase,
        request.index,
        request.incoming,
        request.artifacts.model,
        request.resuming,
    )
    result = run_policy_phase(
        run, request.identity, legacy_recovery=request.legacy_recovery
    ).phases[0]
    return PortablePhaseResult(
        read_execution_artifact(result.provenance["actor_path"]), result.as_dict()
    )
