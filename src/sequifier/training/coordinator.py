"""Atomic phase coordination around the native supervised and policy runners.

The envelope commits an immutable incoming artifact before starting a phase.
Native checkpoints own in-phase optimizer, sampler, and environment recovery.
Completing a phase publishes its selected artifact before advancing the envelope.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from sequifier.artifacts.checkpoint_envelope import (
    checkpoint_envelope,
    recovery_payload,
)
from sequifier.artifacts.model_artifact import (
    ModelArtifact,
    read_execution_artifact,
    save_model_artifact,
)
from sequifier.artifacts.phase import PhaseArtifacts, verify_artifact
from sequifier.artifacts.reports import phase_report, publish_run_report, run_report
from sequifier.fingerprints import fingerprint
from sequifier.io.atomic import atomic_save
from sequifier.runtime.plan import PhaseOutcome, RunIdentity, new_run_id, run_phases
from sequifier.training.phase_runners import (
    PortablePhaseRequest,
    policy_phase_config,
    run_portable_policy_phase,
    run_supervised_phase,
    supervised_phase_config,
)
from sequifier.trajectory.sources import file_checksum


@dataclass(frozen=True)
class LearningRunResult:
    model_path: str
    checkpoint_path: str
    report_path: str
    phases: tuple[dict, ...]


@dataclass(frozen=True)
class PortablePhaseCommit:
    artifacts: PhaseArtifacts
    fingerprint: str
    model: ModelArtifact | None
    report: dict


def coordinate_training(
    config,
    *,
    launch_supervised,
    integration_specs,
    integration_instances,
    semantic_optimizer_grouping,
):
    """Choose native execution and recovery services for the resolved plan."""
    from sequifier.config.train_config import ResolvedSequifierConfig
    from sequifier.io.output_paths import resolve_training_output_paths

    if not isinstance(config, ResolvedSequifierConfig):
        raise TypeError("Training requires a canonical resolved config.")
    resolve_training_output_paths(config)
    phases = config.training_plan
    if all(phase.algorithm == "supervised" for phase in phases):
        return launch_supervised(
            config,
            integration_specs=integration_specs,
            integration_instances=integration_instances,
            semantic_optimizer_grouping=semantic_optimizer_grouping,
        )
    if integration_specs or integration_instances or semantic_optimizer_grouping:
        raise ValueError(
            "policy phases do not support custom integrations or parameter grouping"
        )
    if all(phase.algorithm != "supervised" for phase in phases):
        return run_policy_plan(config)
    return run_learning_plan(config)


def run_policy_plan(config):
    """Use native policy recovery when the plan shares an actor representation."""
    from sequifier.config.training_resolution import policy_run_config
    from sequifier.rl.learner import run_policy_training

    if len({phase.model_interface for phase in config.training_plan}) == 1:
        return run_policy_training(policy_run_config(config, config.training_plan))
    return run_learning_plan(config)


def _legacy_learning_plan_identity(plan):
    """Recover the manifest identity used before runs serialized an interface registry."""
    legacy = dict(plan)
    legacy["dataset_training"] = {
        name: {
            "name": name,
            "model_interface": name,
            "interface": interface,
            "parts": {},
            "criterion": {
                target: "CrossEntropyLoss" if kind == "categorical" else "MSELoss"
                for target, kind in interface["target_column_types"].items()
            },
            "class_weights": None,
            "loss_weights": None,
            "class_share_log_columns": [],
        }
        for name, interface in legacy.pop("interfaces").items()
    }
    legacy.update(curriculum_training=False, curriculum_column=None)
    return fingerprint(legacy)


def _completed_report(item):
    if "report" in item:
        return item["report"]
    artifacts = PhaseArtifacts(Path(item["model_path"]))
    receipt = artifacts.read_receipt(**item["report_receipt"])
    if receipt is None:
        raise ValueError(
            f"Missing committed policy report receipt: {artifacts.receipt}"
        )
    return receipt["report"]


def run_learning_plan(config) -> LearningRunResult:
    phases = config.training_plan
    if any(phase.optimizer_state == "continue" for phase in phases):
        raise ValueError("portable phase handoffs cannot continue optimizer state")
    resume = config.runtime_settings.resume
    from sequifier.config.components import ResumeConfig

    resume = resume or ResumeConfig()
    from sequifier.artifacts.run_checkpoint import checkpoint_path

    checkpoint = checkpoint_path(config)
    plan = config.model_dump(mode="json")
    plan["runtime_settings"].pop("resume", None)
    plan.get("ownership", {}).get("run", {}).get("recovery", {}).pop("resume", None)
    identity = fingerprint(plan)
    state: dict[str, Any] = {
        "artifact_type": "sequifier_learning_run",
        "format_version": 3,
        "plan": identity,
        "completed": [],
        "incoming": config.initialize_from,
        "incoming_sha256": file_checksum(config.initialize_from)
        if config.initialize_from
        else None,
        "active": False,
        "run_id": new_run_id(),
    }
    if resume.policy != "never" and checkpoint.is_file():
        state = recovery_payload(
            torch.load(checkpoint, map_location="cpu", weights_only=False), "learning"
        )
        if (
            state.get("artifact_type") != "sequifier_learning_run"
            or state.get("format_version") not in {2, 3}
            or (
                state.get("plan") != identity
                and state.get("plan") != _legacy_learning_plan_identity(plan)
            )
        ):
            raise ValueError("learning run checkpoint differs from the requested plan")
        state["plan"] = identity
    elif resume.policy == "required":
        raise FileNotFoundError(checkpoint)
    identity_scope = RunIdentity.start(state.get("run_id"))
    state["run_id"] = identity_scope.run_id
    if not 0 <= len(state["completed"]) <= len(phases):
        raise ValueError("learning run checkpoint exceeds the phase plan")
    verify_artifact(
        state["incoming"],
        state["incoming_sha256"],
        "committed incoming model artifact changed",
    )
    initialization = {"path": config.initialize_from}
    if config.actor_initialization == "history_only" and state["completed"]:
        initialization = _completed_report(state["completed"][0])["provenance"][
            "actor_initialization"
        ]
        verify_artifact(
            config.initialize_from,
            initialization["transfer"]["source_sha256"],
            "history initialization artifact changed",
        )
    # Validate model/source/environment contracts for the complete plan first.
    from sequifier.rl.preparation import (
        preflight_environments,
        prepare_data,
        validate_output_paths,
    )
    from sequifier.rl.prepared_inputs import PreparedTrainingInputs

    prepared_inputs = PreparedTrainingInputs.for_run(config)
    dependencies = {}
    protected_inputs = set()
    for index, phase in enumerate(phases):
        if index == 0:
            phase_incoming = config.initialize_from
        elif index <= len(state["completed"]):
            phase_incoming = state["completed"][index - 1]["model_path"]
        else:
            phase_incoming = None
        if phase.algorithm == "supervised":
            resolved = supervised_phase_config(
                config, phase, index, phase_incoming, False
            )
            from sequifier.io.output_paths import supervised_input_paths

            protected_inputs.update(supervised_input_paths(resolved))
            for dataset in resolved.dataset_training.values():
                for part in dataset.parts.values():
                    path = str(Path(config.project_root) / part.metadata_config_path)
                    dependencies[path] = file_checksum(path)
        else:
            destination = PhaseArtifacts.for_phase(
                config.phase_directory, index, phase.name, "model.pt"
            ).model
            run = policy_phase_config(
                config,
                phase,
                index,
                phase_incoming,
                destination,
                False,
            )
            inputs = prepared_inputs.resolve(run)
            inputs.revalidate(inputs.identity, f"phases.{phase.name}.source_identity")
            validate_output_paths(run, inputs)
            dependencies.update(inputs.identity)
            preflight_environments(run)
            prepare_data(run, run.policy, inputs)
            if run.execution_model is None or run.policy_interface is None:
                raise ValueError("a policy phase has no resolved model interface")
    if "dependencies" in state and state["dependencies"] != dependencies:
        raise ValueError("learning plan source identities changed since its checkpoint")
    state["dependencies"] = dependencies
    protected = {Path(path).resolve() for path in dependencies} | protected_inputs
    for phase in phases:
        reference = getattr(phase, "reference", None)
        if reference is not None and reference.path:
            protected.add(Path(reference.path).resolve())
    if config.initialize_from:
        protected.add(Path(config.initialize_from).resolve())
    from sequifier.io.output_paths import validate_training_destinations

    validate_training_destinations(config, checkpoint, protected)

    def save_boundary():
        state["format_version"] = 3
        model = (
            read_execution_artifact(state["incoming"]) if state["incoming"] else None
        )
        if (
            model is not None
            and state["completed"]
            and (state["completed"][-1]["algorithm"] != "supervised")
        ):
            # Older phase models may still embed their policy histories.
            provenance = dict(model.metadata.provenance)
            provenance.pop("metrics", None)
            if "provenance" in provenance:
                provenance["provenance"] = {
                    key: value
                    for key, value in provenance["provenance"].items()
                    if key not in {"evaluations", "continuations"}
                }
            model = replace(
                model, metadata=replace(model.metadata, provenance=provenance)
            )
        atomic_save(checkpoint, checkpoint_envelope("learning", state, model))

    def run_phase(index, phase):
        incoming = state["incoming"]
        verify_artifact(
            incoming,
            state["incoming_sha256"],
            "incoming model artifact changed since the committed boundary",
        )
        artifacts = PhaseArtifacts.for_phase(config.phase_directory, index, phase.name)
        resuming = state["active"] and resume.policy != "never"
        legacy_recovery = resuming and "active_run_id" not in state
        if not state["active"]:
            state.update(active=True, active_run_id=identity_scope.run_id)
            save_boundary()
        phase_identity = fingerprint(
            {
                "phase": phase.model_dump(mode="json"),
                "incoming": state["incoming_sha256"],
            }
        )
        receipt = (
            artifacts.read_receipt(phase_identity, run_id=state.get("active_run_id"))
            if resuming
            else None
        )
        result = None
        if receipt is not None:
            report = receipt["report"]
        else:
            request = PortablePhaseRequest(
                index, incoming, artifacts, resuming, legacy_recovery, identity_scope
            )
            runner = (
                run_supervised_phase
                if phase.algorithm == "supervised"
                else run_portable_policy_phase
            )
            result = runner(config, phase, request)
            details = {
                key: value
                for key, value in result.report.items()
                if key
                not in {
                    "index",
                    "name",
                    "algorithm",
                    "model_path",
                    "checkpoint_path",
                    "artifact_type",
                    "format_version",
                }
            }
            if phase.algorithm != "supervised":
                details["provenance"] = {
                    **details["provenance"],
                    **artifacts.describe(),
                }
            report = phase_report(
                index=index,
                name=phase.name,
                algorithm=phase.algorithm,
                model_path=artifacts.model,
                checkpoint_path=artifacts.model.with_name("run.pt"),
                **details,
            )
        return PhaseOutcome(
            PortablePhaseCommit(
                artifacts, phase_identity, result.artifact if result else None, report
            ),
            report.get("stop_scope"),
        )

    def commit_phase(index, phase, outcome):
        nonlocal initialization
        commit = outcome.result
        artifacts, report = commit.artifacts, commit.report
        if commit.model is not None:
            artifacts.publish_model(commit.model)
            artifacts.write_receipt(
                commit.fingerprint, report, run_id=state.get("active_run_id")
            )
        artifacts.publish_report(report)
        if index == 0 and config.actor_initialization == "history_only":
            initialization = report["provenance"]["actor_initialization"]
        state["stopped"] = outcome.stop_scope == "end_plan"
        checksum = file_checksum(artifacts.model)
        item = {
            "name": phase.name,
            "algorithm": phase.algorithm,
            "model_path": str(artifacts.model),
        }
        if phase.algorithm == "supervised":
            item["report"] = report
        else:
            item["report_receipt"] = {
                "identity": commit.fingerprint,
                "run_id": state.get("active_run_id"),
            }
        state["completed"].append(item)
        state.update(
            incoming=str(artifacts.model),
            incoming_sha256=checksum,
            active=False,
        )
        state.pop("active_run_id", None)
        save_boundary()

    migrated = False
    for index, item in enumerate(state["completed"]):
        artifacts = PhaseArtifacts.for_phase(
            config.phase_directory, index, phases[index].name
        )
        report = _completed_report(item)
        if item["algorithm"] != "supervised" and "report" in item:
            receipt = {"identity": fingerprint(report), "run_id": identity_scope.run_id}
            artifacts.write_receipt(report=report, **receipt)
            item["report_receipt"] = receipt
            del item["report"]
            migrated = True
        artifacts.publish_report(report)
        del report
    if migrated:
        save_boundary()
    completion_reason = run_phases(
        phases,
        run_phase,
        commit_phase,
        start=len(state["completed"]),
        stopped=state.get("stopped", False),
    )
    final = read_execution_artifact(state["incoming"])
    save_model_artifact(final, config.model_path)
    completed = []
    for item in state["completed"]:
        result = dict(item)
        result.pop("report_receipt", None)
        result["report"] = _completed_report(item)
        completed.append(result)
    publish_run_report(
        run_report(
            model_path=config.model_path,
            checkpoint_path=checkpoint,
            phases=[item["report"] for item in completed],
            plan_fingerprint=identity,
            run_id=state["run_id"],
            session_id=identity_scope.session_id,
            source_provenance=dependencies,
            initialization=initialization,
            resume=resume.model_dump(mode="json"),
            exports=[config.model_path],
            status="complete",
            completion_reason=completion_reason,
        ),
        config.report_path,
    )
    return LearningRunResult(
        config.model_path,
        str(checkpoint),
        config.report_path,
        tuple(completed),
    )
