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
from sequifier.config.learning_run import LearningRunConfig
from sequifier.config.train_config import (
    PolicyRunConfig,
    SequifierConfig,
    resolve_sequifier_config,
)
from sequifier.fingerprints import fingerprint
from sequifier.io.atomic import atomic_save, write_json_atomic
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

    model = config.model
    source = None
    forward_attention = None
    if incoming is not None:
        source, _ = resolved_config_from_model_config(
            read_execution_artifact(incoming).model_config.values, device=config.device
        )
        model = source.model
        forward_attention = source.global_training.forward_attention or (
            "bidirectional"
            if source.global_training.training_objective == "bert"
            else "causal"
        )
    training = config.global_training.model_copy(deep=True)
    if training.distributed:
        raise ValueError("mixed learning plans currently require local phase execution")
    if forward_attention is not None:
        training.forward_attention = forward_attention
    from sequifier.config.components import ResumeConfig

    training.resume = ResumeConfig(policy="if_exists" if resume else "never")
    authored = SequifierConfig.model_validate(
        {
            "project_root": config.project_root,
            "model_name": f"{config.model_name}-phase-{index:03d}-{phase.name}",
            "device": config.device,
            "seed": config.seed + index,
            "model": model.model_dump(),
            "global_training": training.model_dump(),
            "dataset_training": {
                name: value.model_dump()
                for name, value in config.dataset_training.items()
            },
            "training_plan": {"phases": [phase.model_dump(exclude={"handoff"})]},
            "evaluation": config.evaluation.model_dump() if config.evaluation else None,
            "export_pt": True,
            "export_onnx": config.export_onnx,
        }
    )
    metadata = {
        f"{name}.{part_name}": load_dataset_metadata(
            str(Path(config.project_root) / part.metadata_config_path)
        )
        for name, dataset in authored.dataset_training.items()
        for part_name, part in dataset.parts.items()
    }
    resolved = resolve_sequifier_config(authored, metadata)
    if source is not None:
        # Preserve policy bindings and their model-owned scale modules on return
        # to supervised learning. Dataset resolution owns the data/window views.
        for dataset in resolved.dataset_training.values():
            previous = source.interface(dataset.model_interface)
            dataset.interface.policy = previous.policy
        from sequifier.artifacts.model_config import execution_config

        existing = {
            dataset.model_interface for dataset in resolved.dataset_training.values()
        }
        for name in source.interface_names:
            if name in existing:
                continue
            interface = source.interface(name)
            placeholder = execution_config(
                source.model, {name: interface}, resolved.global_training
            ).dataset_training[name]
            key = name
            while key in resolved.dataset_training:
                key += "_unused"
            resolved.dataset_training[key] = placeholder.model_copy(
                update={"name": key}
            )
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
        config.seed + index, resolved.global_training.enforce_determinism
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
            "optimizer_steps": run.state.optimizer_step,
            "handoff": phase.handoff,
        }
    finally:
        run.distributed.finalize()


def _policy_config(
    config, phase, index, incoming, destination, resume, execution_model=None
):
    if incoming is None and execution_model is None:
        raise ValueError("a policy phase requires a model execution contract")
    interface = phase.model_interface or config.policy_interface
    if interface is None:
        if execution_model is not None:
            interfaces = execution_model["interfaces"]
        else:
            if incoming is None:
                raise ValueError("policy initialization requires a model artifact")
            interfaces = read_execution_artifact(incoming).model_config.values[
                "interfaces"
            ]
        if len(interfaces) != 1:
            raise ValueError("select policy_interface or a phase model_interface")
        interface = next(iter(interfaces))
    return PolicyRunConfig.model_validate(
        {
            "project_root": config.project_root,
            "model_name": f"{config.model_name}-phase-{index:03d}-{phase.name}",
            "device": config.device,
            "seed": config.seed + index,
            "initialize_from": incoming,
            "policy_interface": interface,
            "execution_model": execution_model,
            "policy_binding": config.policy_bindings[interface].model_dump(
                exclude_unset=True
            )
            if interface in config.policy_bindings
            else None,
            "sources": {
                name: value.model_dump() for name, value in config.sources.items()
            },
            "preferences": config.preferences,
            "units": config.units,
            "phases": [phase.model_dump()],
            "validation_fraction": config.validation_fraction,
            "episode_cache_size": config.episode_cache_size,
            "resume": {
                "policy": "if_exists" if resume else "never",
                "mode": config.resume.mode,
            },
            "model_path": str(destination),
        }
    )


def _run_policy(config, phase, index, incoming, destination, resume):
    from sequifier.rl.learner import run_policy_training

    run = _policy_config(config, phase, index, incoming, destination, resume)
    return run_policy_training(run).phases[0].as_dict()


def run_learning_plan(config: LearningRunConfig) -> LearningRunResult:
    checkpoint = Path(config.checkpoint_path)
    plan = config.model_dump(mode="json", exclude={"resume"})
    identity = fingerprint(plan)
    state: dict[str, Any] = {
        "artifact_type": "sequifier_learning_run",
        "format_version": 1,
        "plan": identity,
        "completed": [],
        "incoming": config.initialize_from,
        "incoming_sha256": file_checksum(config.initialize_from)
        if config.initialize_from
        else None,
        "active": False,
    }
    if config.resume.policy != "never" and checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            state.get("artifact_type") != "sequifier_learning_run"
            or state.get("format_version") != 1
            or state.get("plan") != identity
        ):
            raise ValueError("learning run checkpoint differs from the requested plan")
    elif config.resume.policy == "required":
        raise FileNotFoundError(checkpoint)
    if not 0 <= len(state["completed"]) <= len(config.phases):
        raise ValueError("learning run checkpoint exceeds the phase plan")
    if (
        state["incoming"]
        and file_checksum(state["incoming"]) != state["incoming_sha256"]
    ):
        raise ValueError("committed incoming model artifact changed")
    # Validate model/source/environment contracts for the complete plan first.
    from sequifier.artifacts.model_export import model_execution_config
    from sequifier.rl.preparation import RunInputs, preflight_environments, prepare_data

    execution_model: dict[str, Any] | None = (
        read_execution_artifact(state["incoming"]).model_config.values
        if state["incoming"]
        else None
    )
    dependencies = {}
    for index, phase in enumerate(config.phases):
        if phase.algorithm == "supervised":
            resolved = _supervised_config(
                config, phase, index, state["incoming"], False
            )
            execution_model = model_execution_config(resolved)
            for dataset in resolved.dataset_training.values():
                for part in dataset.parts.values():
                    path = str(Path(config.project_root) / part.metadata_config_path)
                    dependencies[path] = file_checksum(path)
        else:
            destination = (
                Path(config.phase_directory) / f"{index:03d}-{phase.name}" / "model.pt"
            )
            run = _policy_config(
                config,
                phase,
                index,
                state["incoming"],
                destination,
                False,
                execution_model,
            )
            inputs = RunInputs.resolve(run)
            dependencies.update(inputs.identity)
            preflight_environments(run)
            prepare_data(run, run.policy, inputs)
            if run.execution_model is None or run.policy_interface is None:
                raise ValueError("a policy phase has no resolved model interface")
            execution_model = dict(run.execution_model)
            execution_model["interfaces"] = dict(execution_model["interfaces"])
            selected = dict(execution_model["interfaces"][run.policy_interface])
            selected["policy"] = run.policy.model_dump(mode="json")
            execution_model["interfaces"][run.policy_interface] = selected
    if "dependencies" in state and state["dependencies"] != dependencies:
        raise ValueError("learning plan source identities changed since its checkpoint")
    state["dependencies"] = dependencies
    protected = {Path(path).resolve() for path in dependencies}
    if config.initialize_from:
        protected.add(Path(config.initialize_from).resolve())
    outputs = [
        Path(config.checkpoint_path),
        Path(config.model_path),
        Path(config.report_path),
    ]
    if len(set(outputs)) != len(outputs):
        raise ValueError("learning run output paths must be distinct")
    for output in outputs:
        if output in protected or any(
            output in source.parents or source in output.parents for source in protected
        ):
            raise ValueError(f"learning output conflicts with an input: {output}")
    directory = Path(config.phase_directory)
    if any(directory == source or directory in source.parents for source in protected):
        raise ValueError("phase_directory overlaps a learning input")
    for index in range(len(state["completed"]), len(config.phases)):
        phase = config.phases[index]
        incoming = state["incoming"]
        if incoming and file_checksum(incoming) != state["incoming_sha256"]:
            raise ValueError(
                "incoming model artifact changed since the committed boundary"
            )
        destination = (
            Path(config.phase_directory) / f"{index:03d}-{phase.name}" / "model.pt"
        )
        resuming = state["active"] and config.resume.policy != "never"
        if not state["active"]:
            state["active"] = True
            atomic_save(checkpoint, state)
        receipt_path = destination.with_name("completed.json")
        import json

        receipt = (
            json.loads(receipt_path.read_text())
            if resuming and receipt_path.is_file()
            else None
        )
        phase_identity = fingerprint(
            {
                "phase": phase.model_dump(mode="json"),
                "incoming": state["incoming_sha256"],
            }
        )
        if receipt is not None:
            if receipt["identity"] != phase_identity or receipt[
                "sha256"
            ] != file_checksum(destination):
                raise ValueError("completed phase artifact or receipt changed")
            report = receipt["report"]
        else:
            runner = _run_supervised if phase.algorithm == "supervised" else _run_policy
            report = runner(config, phase, index, incoming, destination, resuming)
            write_json_atomic(
                receipt_path,
                {
                    "identity": phase_identity,
                    "sha256": file_checksum(destination),
                    "report": report,
                },
            )
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
        config.checkpoint_path,
        config.report_path,
        tuple(state["completed"]),
    )
