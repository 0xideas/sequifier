"""Coordinate native training and commit portable handoffs."""

from dataclasses import dataclass
from os import link
from pathlib import Path
from shutil import copyfile
from uuid import uuid4

import torch

from sequifier.artifacts.backbone_repository import select_revision
from sequifier.artifacts.checkpoint_envelope import (
    checkpoint_envelope,
    phase_boundary,
    recovery_payload,
    require_phase_boundary,
)
from sequifier.artifacts.dependencies import (
    DependencyManifest,
    add_backbone_revision,
    add_initialization,
    supervised_dependencies,
)
from sequifier.artifacts.model_artifact import (
    read_execution_artifact,
    save_model_artifact,
)
from sequifier.artifacts.phase import (
    PhaseArtifacts,
    PhaseCommit,
    publish_phase_commits,
    verify_artifact,
)
from sequifier.artifacts.reports import publish_run_report, run_report
from sequifier.artifacts.run_checkpoint import checkpoint_path, training_plan_identity
from sequifier.config.components import ResumeConfig
from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.io.atomic import atomic_path, atomic_save
from sequifier.io.output_paths import validate_training_destinations
from sequifier.runtime.plan import (
    PhaseOutcome,
    RunIdentity,
    continuation_groups,
    new_run_id,
    run_phases,
)
from sequifier.training.phase_runners import (
    PortableGroup,
    policy_group_config,
    run_portable_policy_group,
    run_supervised_group,
    supervised_group_config,
)
from sequifier.trajectory.sources import file_checksum


@dataclass(frozen=True)
class LearningRunResult:
    model_path: str
    checkpoint_path: str
    report_path: str
    phases: tuple[dict, ...]


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

    if not isinstance(config, ResolvedSequifierConfig):
        raise TypeError("Training requires a canonical resolved config.")
    if config.publication.mode == "supervised":
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
    return run_policy_plan(config)


def run_policy_plan(config):
    """Use native policy recovery when the plan shares an actor representation."""
    from sequifier.config.training_resolution import policy_run_config
    from sequifier.rl.learner import run_policy_training
    from sequifier.rl.prepared_inputs import PreparedTrainingInputs

    if config.publication.mode == "policy":
        return run_policy_training(
            policy_run_config(config, config.training_plan),
            prepared=PreparedTrainingInputs.for_run(config),
        )
    return run_learning_plan(config)


def _snapshot_recovery(source, destination):
    """Pin an atomic native checkpoint without depending on its rolling path."""
    with atomic_path(destination, durable=True) as temporary:
        try:
            link(source, temporary)
        except OSError:
            copyfile(source, temporary)


def _portable_commit(config, index, phase, checkpoint, values):
    commit = PhaseCommit(**values)
    if commit.result["index"] != index:
        raise ValueError("native phase commit disagrees with the original phase index")
    artifacts = PhaseArtifacts.for_phase(config.phase_directory, index, phase.name)
    report = {
        **commit.result,
        "model_path": str(artifacts.model),
        "checkpoint_path": str(checkpoint),
        "provenance": {**commit.result["provenance"], **artifacts.describe()},
    }
    if phase.algorithm == "supervised":
        report["optimizer_steps"] = report["clocks"]["optimizer_steps"]
    return commit.with_result(report)


def run_learning_plan(config) -> LearningRunResult:
    from sequifier.rl.prepared_inputs import PreparedTrainingInputs

    prepared = PreparedTrainingInputs.for_run(config)
    with prepared.run_scope():
        return _run_learning_plan(config, prepared)


def _prepare_plan(config, prepared, groups, revision, state):
    from sequifier.rl.preparation import PolicyGroupPreparation
    from sequifier.training.runtime import preflight_supervised_data

    manifest = DependencyManifest()
    add_initialization(manifest, config)
    add_backbone_revision(
        manifest, config, revision["revision_id"] if revision else None
    )
    last_data_phase = -1
    supervised_runs = []
    policy_groups = {}
    committed = len(state["completed"])
    active = state["active"]
    for index, phases in groups:
        phase = phases[0]
        if phase.algorithm == "supervised":
            native = supervised_group_config(config, phases, index, False)
            manifest.merge(supervised_dependencies(native))
            preflight_supervised_data(native)
            supervised_runs.append((index, native))
            continue
        native = policy_group_config(config, phases, index)
        # Preflight can defer the active reference to the native checkpoint.
        # The policy runner verifies that its weights were actually embedded.
        can_recover_active = (
            active is not None
            and active["index"] == index
            and active["recovery"] is not None
        )
        preparation = PolicyGroupPreparation.build(
            native,
            prepared,
            native.policy,
            saved=state.get("dependencies"),
            committed_reference_count=max(
                0, committed - index + int(can_recover_active)
            ),
        )
        manifest.merge(preparation.dependencies)
        if index + len(phases) > committed:
            policy_groups[index] = preparation
        for offset, part in enumerate(preparation.data):
            if part is not None:
                last_data_phase = index + offset
    for index, native in supervised_runs:
        validate_training_destinations(
            native,
            checkpoint_path(native),
            manifest.protected_paths,
            phase_offset=index,
        )
    return manifest, last_data_phase, policy_groups


def _run_learning_plan(config, prepared):
    phases = config.training_plan
    groups = [
        (index, tuple(group))
        for index, group in continuation_groups(phases, group_supervised=True)
    ]
    resume = config.runtime_settings.resume or ResumeConfig()
    checkpoint = checkpoint_path(config)
    plan_identity = training_plan_identity(config)
    state = {
        "artifact_type": "sequifier_learning_run",
        "format_version": ContractVersions.LEARNING_RUN_STATE,
        "plan": plan_identity,
        "completed": [],
        "active": None,
        "run_id": new_run_id(),
    }
    envelope = None
    resuming = resume.policy != "never" and checkpoint.is_file()
    if resuming:
        envelope = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = recovery_payload(envelope, "learning")
        require_contract_version(
            "learning run coordinator state",
            state.get("format_version"),
            ContractVersions.LEARNING_RUN_STATE,
        )
        if (
            state.get("artifact_type") != "sequifier_learning_run"
            or state.get("plan") != plan_identity
        ):
            raise ValueError("learning run checkpoint differs from the requested plan")
    elif resume.policy == "required":
        raise FileNotFoundError(checkpoint)
    state["seed"] = config.seed

    revision = None
    if (
        phases[0].algorithm == "supervised"
        and not config.initialize_from
        and (not resuming or state["backbone_revision"] is not None)
    ):
        revision = select_revision(
            config.model.backbone,
            config.project_root,
            revision_id=state.get("backbone_revision"),
        )
    state["backbone_revision"] = revision["revision_id"] if revision else None
    identity = RunIdentity.start(state["run_id"])
    completed = [PhaseCommit.load(values) for values in state["completed"]]
    if len(completed) > len(phases) or any(
        (commit.result["index"], commit.result["name"], commit.result["algorithm"])
        != (index, phase.name, phase.algorithm)
        for index, (commit, phase) in enumerate(zip(completed, phases))
    ):
        raise ValueError("learning run checkpoint differs from the phase plan")
    active = state["active"]
    if active is not None:
        ends = {start: start + len(group) for start, group in groups}
        start = active["index"]
        if (
            active["run_id"] != identity.run_id
            or start not in ends
            or not start <= len(completed) < ends[start]
            or active["phase_index"] != len(completed)
            or (active["recovery"] is None and start != len(completed))
        ):
            raise ValueError(
                "learning run checkpoint has an invalid active phase group"
            )
    manifest, last_data_phase, policy_groups = _prepare_plan(
        config, prepared, groups, revision, state
    )
    if resuming:
        manifest.validate(state["dependencies"])
    state["dependencies"] = manifest.identity
    validate_training_destinations(config, checkpoint, manifest.protected_paths)

    if last_data_phase < len(completed):
        prepared.release_data()

    def artifact_before(index):
        return (
            completed[index - 1].model_path if index else config.initialize_from or None
        )

    initial = config.initialize_from
    initial_identity = (
        {"path": initial, "sha256": manifest.identity[str(Path(initial).resolve())]}
        if initial
        else None
    )

    def boundary():
        return phase_boundary(
            identity.run_id,
            selected=completed[-1].model_identity if completed else initial_identity,
            active=state["active"],
        )

    if envelope is not None:
        require_phase_boundary(envelope, boundary())

    selected_path, selected_model = None, None

    def save_boundary():
        nonlocal selected_path, selected_model
        selected = artifact_before(len(completed))
        if selected != selected_path:
            selected_model = (
                read_execution_artifact(selected) if selected is not None else None
            )
            selected_path = selected
        atomic_save(
            checkpoint,
            checkpoint_envelope(
                "learning",
                state,
                selected_model,
                boundary=boundary(),
            ),
        )

    def run_group(transition):
        index, group = transition.phase
        first = group[0]
        base = PhaseArtifacts.for_phase(config.phase_directory, index, first.name)
        native_checkpoint = base.model.with_name("run.pt")
        manifest.revalidate()
        if index:
            incoming = completed[index - 1]
            verify_artifact(
                incoming.model_path,
                incoming.model_sha256,
                "Selected phase model changed",
            )
        active = state["active"]
        if active is None:
            active = {
                "index": index,
                "phase_index": index,
                "run_id": identity.run_id,
                "recovery": None,
            }
            state["active"] = active
            save_boundary()
        prior = active["recovery"]
        if prior is not None:
            verify_artifact(
                prior["path"], prior["sha256"], "Active phase recovery changed"
            )
            _snapshot_recovery(prior["path"], native_checkpoint)

        def save_progress(path, records):
            previous = active["recovery"]
            first_new = len(completed) - index
            new = [
                _portable_commit(
                    config, index + offset, group[offset], native_checkpoint, values
                )
                for offset, values in enumerate(records[first_new:], first_new)
            ]
            finished = len(records) == len(group) or (
                new and new[-1].result["stop_scope"] == "end_plan"
            )
            if finished:
                state["active"] = None
            else:
                recovery_path = (
                    base.model.parent / "commits" / f"{uuid4().hex}.recovery.pt"
                )
                _snapshot_recovery(path, recovery_path)
                active["phase_index"] = len(completed) + len(new)
                active["recovery"] = {
                    "path": str(recovery_path),
                    "sha256": file_checksum(recovery_path),
                }
            completed.extend(new)
            state["completed"].extend(commit.as_dict() for commit in new)
            save_boundary()
            publish_phase_commits(new)
            if previous is not None:
                Path(previous["path"]).unlink(missing_ok=True)

        request = PortableGroup(
            index,
            group,
            artifact_before(index),
            prior is not None,
            identity,
            save_progress,
        )
        if first.algorithm == "supervised":
            run_supervised_group(config, request, revision if revision else {})
        else:
            run_portable_policy_group(
                config, request, prepared, policy_groups.pop(index)
            )
        stop = completed[-1].result["stop_scope"]
        if stop != "end_plan" and len(completed) != index + len(group):
            raise ValueError("native phase group returned before committing its phases")
        return PhaseOutcome(None, stop)

    def finish_group(transition, outcome):
        if transition.phase[0] <= last_data_phase < len(completed):
            prepared.release_data()

    next_index = state["active"]["index"] if state["active"] else len(completed)
    start = next(
        (offset for offset, (index, _) in enumerate(groups) if index == next_index),
        len(groups),
    )
    stopped = bool(completed and completed[-1].result["stop_scope"] == "end_plan")
    if start == len(groups) and next_index != len(phases) and not stopped:
        raise ValueError("learning run checkpoint is between inconsistent phase groups")
    completion_reason = run_phases(
        groups,
        run_group,
        finish_group,
        start=start,
        stopped=stopped and state["active"] is None,
        replay=lambda: publish_phase_commits(completed),
    )
    final = read_execution_artifact(completed[-1].model_path)
    save_model_artifact(final, config.model_path)
    reports = [commit.report_result() for commit in completed]
    initialization = reports[0]["provenance"].get(
        "actor_initialization", {"path": config.initialize_from}
    )
    prepared.close()
    publish_run_report(
        run_report(
            model_path=config.model_path,
            checkpoint_path=checkpoint,
            phases=reports,
            plan_fingerprint=plan_identity,
            run_id=identity.run_id,
            session_id=identity.session_id,
            source_provenance=manifest.identity,
            initialization=initialization,
            resume=resume.model_dump(mode="json"),
            exports=[config.model_path],
            status="complete",
            completion_reason=completion_reason,
            resource_accounting=prepared.accounting.snapshot(),
        ),
        config.report_path,
    )
    return LearningRunResult(
        config.model_path,
        str(checkpoint),
        config.report_path,
        tuple(
            {
                **{key: report[key] for key in ("name", "algorithm", "model_path")},
                "report": report,
            }
            for report in reports
        ),
    )
