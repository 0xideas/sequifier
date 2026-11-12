"""Atomic phase coordination around the native supervised and policy runners.

The envelope commits an immutable incoming artifact before starting a phase.
Native checkpoints own in-phase optimizer, sampler, and environment recovery.
Completing a phase publishes its selected artifact before advancing the envelope.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sequifier.artifacts.model_artifact import (
    build_model_artifact,
    read_execution_artifact,
    save_model_artifact,
)
from sequifier.artifacts.phase import PhaseArtifacts, verify_artifact
from sequifier.config.train_config import SequifierConfig, resolve_sequifier_config
from sequifier.fingerprints import fingerprint
from sequifier.io.atomic import atomic_save, write_json_atomic
from sequifier.io.output_paths import (
    contains_path,
    unique_paths,
    validate_path_conflicts,
)
from sequifier.trajectory.sources import file_checksum


@dataclass(frozen=True)
class LearningRunResult:
    model_path: str
    checkpoint_path: str
    report_path: str
    phases: tuple[dict, ...]


def _supervised_config(config, phase, index, incoming, resume):
    from sequifier.artifacts.model_config import resolved_config_from_model_config
    from sequifier.config.metadata import load_dataset_metadata

    authored = config.authored
    # The coordinator stores resolved phases; the native builder owns source expansion.
    phase_spec = authored.training_plan.phases[index].model_copy(deep=True)
    phase_spec.optimization = phase.optimization
    model = config.model
    source = None
    forward_attention = None
    if incoming is not None:
        source, _ = resolved_config_from_model_config(
            read_execution_artifact(incoming).model_config.values, device=config.device
        )
        from sequifier.artifacts.model_config import extend_prediction_distributions

        extend_prediction_distributions(source, config.model)
        model = source.model
        forward_attention = source.runtime_settings.forward_attention or (
            "bidirectional"
            if source.runtime_settings.training_objective == "bert"
            else "causal"
        )
    training = config.runtime_settings.model_copy(deep=True)
    if training.distributed:
        raise ValueError("mixed learning plans currently require local phase execution")
    if forward_attention is not None:
        training.forward_attention = forward_attention
    from sequifier.config.components import ResumeConfig

    training.resume = ResumeConfig(policy="if_exists" if resume else "never")
    values = authored.model_dump(mode="python")
    run = values["run"]
    run["project_root"] = config.project_root
    run["model_name"] = f"{config.model_name}-phase-{index:03d}-{phase.name}"
    run["execution"]["device"] = config.device
    run["reproducibility"]["seed"] = config.seed + index
    run["recovery"]["resume"] = training.resume.model_dump()
    run["export"]["pt"] = True
    model_values = model.model_dump()
    model_values["execution"] = {
        key: getattr(training, key)
        for key in (
            "context_length",
            "training_objective",
            "target_offset",
            "forward_attention",
            "next_occurrence_config",
        )
    }
    if training.next_occurrence_config is not None:
        model_values["execution"]["next_occurrence_config"] = (
            training.next_occurrence_config.model_dump()
        )
    for name, interface in authored.model.interfaces.items():
        if name in model_values["interfaces"]:
            model_values["interfaces"][name]["input_contract"] = (
                interface.input_contract.model_dump()
            )
    values["model"] = model_values
    names = {ref.dataset for ref in [*phase.sources, *phase.evaluation_sources]}
    values["dataset_training"] = {
        name: value
        for name, value in values["dataset_training"].items()
        if name in names
    }
    values["training_plan"] = {"phases": [phase_spec.model_dump()]}
    values["initialize_from"] = None
    authored = SequifierConfig.model_validate(values)
    metadata = {
        f"{name}.{part_name}": load_dataset_metadata(
            str(Path(config.project_root) / part.metadata_config_path)
        )
        for name, dataset in authored.dataset_training.items()
        for part_name, part in dataset.parts.items()
    }
    resolved = resolve_sequifier_config(authored, metadata)
    resolved.ownership = config.ownership
    resolved.source_validation_fractions = config.source_validation_fractions
    resolved.normalization_by_interface = config.normalization_by_interface
    if source is not None:
        # Preserve policy bindings and their model-owned scale modules on return
        # to supervised learning. Dataset resolution owns the data/window views.
        for dataset in resolved.dataset_training.values():
            previous = source.interface(dataset.model_interface)
            dataset.interface.policy = previous.policy
        from sequifier.model.execution import ModelExecution

        ModelExecution.from_config(source).retain_interfaces(resolved)
    if incoming is not None:
        from sequifier.artifacts.model_config import validate_execution_attachment
        from sequifier.artifacts.model_export import model_execution_config

        validate_execution_attachment(
            read_execution_artifact(incoming).model_config.values,
            model_execution_config(resolved),
        )
    return resolved


def _run_supervised(config, phase, index, incoming, destination, resume):
    from sequifier.helpers import configure_determinism, configure_logger
    from sequifier.integration import IntegrationManager
    from sequifier.runtime.builder import RunBuilder
    from sequifier.runtime.context import ExecutionEnvironment
    from sequifier.training.engine import TrainingEngine

    resolved = _supervised_config(config, phase, index, incoming, resume)
    configure_logger(config.project_root, resolved.model_name, 0)
    configure_determinism(
        config.seed + index, resolved.runtime_settings.enforce_determinism
    )
    environment = ExecutionEnvironment(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device(config.device),
        distributed=False,
    )
    integrations = IntegrationManager(rank=0, world_size=1, distributed=False)
    run = RunBuilder().build(
        resolved, environment, integrations, initial_artifact=incoming
    )
    try:
        result = TrainingEngine().run(run)
        if phase.handoff == "best" and run.state.best_model_state_dict:
            run.network.load_state_dict(run.state.best_model_state_dict, strict=True)
        save_model_artifact(build_model_artifact(run.network, resolved), destination)
        return {
            "completion_reason": result.completion_reason,
            "resolved_seed": config.seed + index,
            "stop_scope": phase.stopping.on_trigger
            if run.state.stopping_state.get("triggered")
            else None,
            "optimizer_steps": run.state.optimizer_step,
            "handoff": phase.handoff,
        }
    finally:
        run.distributed.finalize()


def _policy_config(config, phase, index, incoming, destination, resume):
    from sequifier.config.training_resolution import policy_run_config

    return policy_run_config(
        config,
        [phase],
        incoming=incoming,
        index=index,
        destination=destination,
        resuming=resume,
    )


def _run_policy(config, phase, index, incoming, destination, resume):
    from sequifier.rl.learner import run_policy_training

    run = _policy_config(config, phase, index, incoming, destination, resume)
    return run_policy_training(run).phases[0].as_dict()


def run_learning_plan(config) -> LearningRunResult:
    phases = config.training_plan
    resume = config.runtime_settings.resume
    from sequifier.config.components import ResumeConfig

    resume = resume or ResumeConfig()
    checkpoint = Path(config.project_root) / (
        resume.checkpoint_path or f"checkpoints/{config.model_name}/learning-run.pt"
    )
    plan = config.model_dump(mode="json")
    plan["runtime_settings"].pop("resume", None)
    plan.get("ownership", {}).get("run", {}).get("recovery", {}).pop("resume", None)
    identity = fingerprint(plan)
    state: dict[str, Any] = {
        "artifact_type": "sequifier_learning_run",
        "format_version": 2,
        "plan": identity,
        "completed": [],
        "incoming": config.initialize_from,
        "incoming_sha256": file_checksum(config.initialize_from)
        if config.initialize_from
        else None,
        "active": False,
    }
    if resume.policy != "never" and checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            state.get("artifact_type") != "sequifier_learning_run"
            or state.get("format_version") != 2
            or state.get("plan") != identity
        ):
            raise ValueError("learning run checkpoint differs from the requested plan")
    elif resume.policy == "required":
        raise FileNotFoundError(checkpoint)
    if not 0 <= len(state["completed"]) <= len(phases):
        raise ValueError("learning run checkpoint exceeds the phase plan")
    verify_artifact(
        state["incoming"],
        state["incoming_sha256"],
        "committed incoming model artifact changed",
    )
    # Validate model/source/environment contracts for the complete plan first.
    from sequifier.rl.preparation import preflight_environments, prepare_data
    from sequifier.rl.prepared_inputs import PreparedTrainingInputs

    prepared_inputs = PreparedTrainingInputs.for_run(config)
    dependencies = {}
    for index, phase in enumerate(phases):
        if phase.algorithm == "supervised":
            resolved = _supervised_config(
                config, phase, index, state["incoming"], False
            )
            for dataset in resolved.dataset_training.values():
                for part in dataset.parts.values():
                    path = str(Path(config.project_root) / part.metadata_config_path)
                    dependencies[path] = file_checksum(path)
        else:
            destination = PhaseArtifacts.for_phase(
                config.phase_directory, index, phase.name, "model.pt"
            ).model
            run = _policy_config(
                config,
                phase,
                index,
                state["incoming"],
                destination,
                False,
            )
            inputs = prepared_inputs.resolve(run)
            inputs.revalidate(inputs.identity, f"phases.{phase.name}.source_identity")
            dependencies.update(inputs.identity)
            preflight_environments(run)
            prepare_data(run, run.policy, inputs)
            if run.execution_model is None or run.policy_interface is None:
                raise ValueError("a policy phase has no resolved model interface")
    if "dependencies" in state and state["dependencies"] != dependencies:
        raise ValueError("learning plan source identities changed since its checkpoint")
    state["dependencies"] = dependencies
    protected = {Path(path).resolve() for path in dependencies}
    if config.initialize_from:
        protected.add(Path(config.initialize_from).resolve())
    outputs = unique_paths(
        [
            checkpoint,
            Path(config.model_path),
            Path(config.report_path),
        ],
        "learning run output paths must be distinct",
    )
    validate_path_conflicts(
        outputs, protected, label="learning output", check_each_other=False
    )
    directory = Path(config.phase_directory).resolve()
    if any(contains_path(directory, source) for source in protected):
        raise ValueError("phase_directory overlaps a learning input")
    for index in range(len(state["completed"]), len(phases)):
        if state.get("stopped"):
            break
        phase = phases[index]
        incoming = state["incoming"]
        verify_artifact(
            incoming,
            state["incoming_sha256"],
            "incoming model artifact changed since the committed boundary",
        )
        artifacts = PhaseArtifacts.for_phase(
            config.phase_directory, index, phase.name, "model.pt"
        )
        destination = artifacts.model
        resuming = state["active"] and resume.policy != "never"
        if not state["active"]:
            state["active"] = True
            atomic_save(checkpoint, state)
        phase_identity = fingerprint(
            {
                "phase": phase.model_dump(mode="json"),
                "incoming": state["incoming_sha256"],
            }
        )
        receipt = artifacts.read_receipt(phase_identity) if resuming else None
        if receipt is not None:
            report = receipt["report"]
        else:
            runner = _run_supervised if phase.algorithm == "supervised" else _run_policy
            report = runner(config, phase, index, incoming, destination, resuming)
            artifacts.write_receipt(phase_identity, report)
        state["stopped"] = report.get("stop_scope") == "end_plan"
        state["completed"].append(
            {
                "name": phase.name,
                "algorithm": phase.algorithm,
                "model_path": str(destination),
                "report": report,
            }
        )
        state.update(
            incoming=str(destination),
            incoming_sha256=file_checksum(destination),
            active=False,
        )
        atomic_save(checkpoint, state)
    final = read_execution_artifact(state["incoming"])
    save_model_artifact(final, config.model_path)
    write_json_atomic(
        config.report_path,
        {
            "format": "sequifier.learning-run-report",
            "version": 1,
            "plan_fingerprint": identity,
            "phases": state["completed"],
            "model_path": config.model_path,
        },
    )
    return LearningRunResult(
        config.model_path,
        str(checkpoint),
        config.report_path,
        tuple(state["completed"]),
    )
