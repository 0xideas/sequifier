"""Coordinate native training and commit portable handoffs."""

from __future__ import annotations

from dataclasses import dataclass
from os import link
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import torch

from sequifier.artifacts.backbone_repository import (
    publish_selected_backbone,
    select_revision,
)
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
)
from sequifier.artifacts.model_artifact import read_execution_artifact
from sequifier.artifacts.phase import (
    CommittedPhases,
    PhaseCommit,
    PhasePublication,
    publish_phase_commits,
    verify_artifact,
)
from sequifier.artifacts.reports import (
    completed_run_report,
    publish_run_report,
    run_report,
)
from sequifier.artifacts.run_checkpoint import (
    checkpoint_path,
    select_run_checkpoint,
    training_plan_identity,
)
from sequifier.config.components import ResumeConfig
from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.fingerprints import file_checksum
from sequifier.io.atomic import atomic_path, atomic_save
from sequifier.io.output_paths import policy_graph_paths, validate_training_destinations
from sequifier.runtime.plan import (
    PhaseOutcome,
    RunIdentity,
    continuation_groups,
    new_run_id,
    run_phases,
)
from sequifier.training.metrics import (
    model_metric_tables,
    progress_after,
    restore_metric_positions,
)
from sequifier.training.phase_runners import (
    PolicyPhaseGroup,
    SupervisedPhaseGroup,
    compile_phase_group,
)
from sequifier.training.rollout_sources import PendingGroup

if TYPE_CHECKING:
    from sequifier.config.train_config import ResolvedSequifierConfig
    from sequifier.integration import IntegrationSpec
    from sequifier.rl.run_state import PolicyRunResult


@dataclass(frozen=True)
class LearningRunResult:
    model_path: str
    checkpoint_path: str
    report_path: str
    phases: tuple[dict, ...]


def run_training(
    config: ResolvedSequifierConfig,
    *,
    integration_specs: tuple[IntegrationSpec, ...] = (),
    integration_instances: tuple[Any, ...] = (),
    semantic_optimizer_grouping: bool = False,
) -> LearningRunResult | PolicyRunResult | None:
    """Choose native execution and recovery services for the resolved plan."""
    from sequifier.config.train_config import ResolvedSequifierConfig

    if not isinstance(config, ResolvedSequifierConfig):
        raise TypeError("Training requires a canonical resolved config.")
    if config.run_mode != "supervised" and (
        integration_specs or integration_instances or semantic_optimizer_grouping
    ):
        raise ValueError(
            "policy phases do not support custom integrations or parameter grouping"
        )
    from sequifier.training.launch import launch_training

    return launch_training(
        config,
        integration_specs=integration_specs,
        integration_instances=integration_instances,
        semantic_optimizer_grouping=semantic_optimizer_grouping,
    )


def run_policy_plan(config) -> LearningRunResult | PolicyRunResult | None:
    return run_training(config)


def execute_policy_plan(config, environment):
    """Use native policy recovery when the plan shares an actor representation."""
    from sequifier.config.training_resolution import policy_run_config
    from sequifier.rl.learner import execute_policy
    from sequifier.rl.prepared_inputs import PreparedTrainingInputs

    prepared = PreparedTrainingInputs.for_run(config)
    with prepared.run_scope():
        if config.run_mode == "policy":
            native = policy_run_config(config, config.training_plan)
            return execute_policy(native, prepared, environment=environment)
        return _run_learning_plan(config, prepared, environment)


def _snapshot_recovery(source, destination):
    """Pin an atomic native checkpoint without depending on its rolling path."""
    with atomic_path(destination, durable=True) as temporary:
        try:
            link(source, temporary)
        except OSError:
            copyfile(source, temporary)


def _prepare_plan(config, groups, revision, state, completed):
    manifest = DependencyManifest()
    add_initialization(manifest, config)
    add_backbone_revision(
        manifest, config, revision["revision_id"] if revision else None
    )
    committed = len(completed)
    stopped = bool(completed and completed[-1].result["stop_scope"] == "end_plan")
    last_data_phase = -1
    active = state["active"]
    for group in groups:
        if isinstance(group, PendingGroup):
            continue
        # A recoverable active phase can use the reference embedded in its checkpoint.
        recovering = (
            active is not None
            and active["index"] == group.index
            and active["recovery"] is not None
        )
        dependencies, last = group.preflight(
            state.get("dependencies"),
            max(0, committed - group.index + int(recovering)),
            finished=stopped or group.index + len(group.phases) <= committed,
        )
        manifest.merge(dependencies)
        last_data_phase = max(last_data_phase, last)
        if group.index + len(group.phases) <= committed:
            group.release()
    for group in groups:
        if isinstance(group, PendingGroup):
            continue
        validate_training_destinations(
            group.config,
            group.artifacts.recovery,
            manifest.protected_paths,
            phase_offset=group.index,
            group=True,
        )
    return manifest, last_data_phase


def _run_learning_plan(config, prepared, environment):
    from sequifier.training.distributed_strategy import on_writer, strategy_for

    strategy = strategy_for(environment, config.run.execution.distribution_settings)
    writer = strategy.rank == 0
    resume = config.run.recovery.resume or ResumeConfig()
    checkpoint = checkpoint_path(config)
    state = {
        "artifact_type": "sequifier_learning_run",
        "format_version": ContractVersions.LEARNING_RUN_STATE,
        "completed": [],
        "active": None,
        "run_id": new_run_id(),
    }
    envelope = None
    selected_checkpoint = select_run_checkpoint(checkpoint, resume)
    resuming = selected_checkpoint is not None
    if selected_checkpoint is not None:
        envelope = torch.load(
            selected_checkpoint, map_location="cpu", weights_only=False
        )
        state = recovery_payload(envelope, "learning")
        require_contract_version(
            "learning run coordinator state",
            state.get("format_version"),
            ContractVersions.LEARNING_RUN_STATE,
        )
        if state.get("artifact_type") != "sequifier_learning_run":
            raise ValueError("learning run checkpoint differs from the requested plan")
    state["run_id"] = strategy.broadcast(state["run_id"])
    active = state["active"]
    # Policy groups retain this parent, including resolved supervised scheduler budgets.
    config = config.model_copy()
    groups: list[PolicyPhaseGroup | SupervisedPhaseGroup | PendingGroup] = [
        compile_phase_group(config, phases, index, prepared)
        for index, phases in continuation_groups(
            config.training_plan, group_supervised=True
        )
    ]
    config.training_plan = [phase for group in groups for phase in group.phases]
    plan_identity = training_plan_identity(config)
    if config._rollout_plan is not None:
        from sequifier.fingerprints import fingerprint

        groups.extend(PendingGroup.planned(config, len(config.training_plan)))
        authored = config._rollout_plan.model_dump(
            mode="json", exclude={"run": {"recovery", "retrain"}}
        )
        if authored["run"]["execution"].get("distribution") is None:
            authored["run"]["execution"].pop("distribution", None)
        plan_identity = fingerprint(
            {"resolved_prefix": plan_identity, "authored_plan": authored}
        )
    phases = [phase for group in groups for phase in group.phases]
    if resuming and state.get("plan") != plan_identity:
        raise ValueError("learning run checkpoint differs from the requested plan")
    state["plan"] = plan_identity
    state["seed"] = config.seed

    revision = None
    if not config.initialize_from and (
        not resuming or state["backbone_revision"] is not None
    ):
        revision = on_writer(
            strategy,
            lambda: select_revision(
                config.execution.backbone,
                config.project_root,
                revision_id=state.get("backbone_revision"),
            ),
        )
    state["backbone_revision"] = revision["revision_id"] if revision else None
    identity = strategy.broadcast(RunIdentity.start(state["run_id"]))
    completed = CommittedPhases.load(phases, state.pop("completed"))
    if active is not None:
        ends = {group.index: group.index + len(group.phases) for group in groups}
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
    manifest, last_data_phase = _prepare_plan(
        config, groups, revision, state, completed
    )
    if resuming:
        manifest.validate(state["dependencies"])
    state["dependencies"] = manifest.identity
    for recorded in state.get("deferred_dependencies", {}).values():
        saved_inputs = DependencyManifest()
        saved_inputs.include(recorded)
        saved_inputs.revalidate()
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
            selected=completed.selected_model or initial_identity,
            active=state["active"],
        )

    if envelope is not None:
        require_phase_boundary(envelope, boundary())
        if writer and "metric_positions" in state:
            restore_metric_positions(
                model_metric_tables(config.project_root, config.model_name),
                identity.run_id,
                state["metric_positions"],
                missing_offset=0,
            )

    selected_path, selected_model = None, None

    def save_boundary():
        nonlocal selected_path, selected_model
        if not writer:
            return
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
                {
                    **state,
                    "completed": completed.records(),
                    "metric_positions": {
                        path.name: path.stat().st_size
                        for path, _ in model_metric_tables(
                            config.project_root, config.model_name
                        )
                    },
                },
                selected_model,
                boundary=boundary(),
            ),
        )

    def run_group(transition):
        group = transition.phase
        if isinstance(group, PendingGroup):
            group, inputs = group.compile(
                config, completed, artifact_before(group.index), prepared, strategy
            )
            recovering = (
                state["active"] is not None and state["active"]["recovery"] is not None
            )
            deps, _ = group.preflight(
                state.get("deferred_dependencies", {}).get(str(group.index)),
                len(completed) - group.index + int(recovering),
            )
            inputs.merge(deps)
            expected = state.setdefault("deferred_dependencies", {}).setdefault(
                str(group.index), inputs.identity
            )
            inputs.validate(expected)
            inputs.revalidate()
            validate_training_destinations(
                group.config,
                group.artifacts.recovery,
                inputs.protected_paths,
                phase_offset=group.index,
                group=True,
            )
        index = group.index
        base = group.artifacts
        native_checkpoint = base.recovery
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
            if writer:
                _snapshot_recovery(prior["path"], native_checkpoint)
            strategy.barrier()

        def save_progress(path, commit: PhaseCommit | None):
            previous = active["recovery"]
            if commit is not None:
                completed.append(commit)
            if len(completed) == index + len(group.phases) or (
                commit is not None and commit.result["stop_scope"] == "end_plan"
            ):
                state["active"] = None
            else:
                recovery_path = base.commits / f"{uuid4().hex}.recovery.pt"
                _snapshot_recovery(path, recovery_path)
                active["phase_index"] = len(completed)
                active["recovery"] = {
                    "path": str(recovery_path),
                    "sha256": file_checksum(recovery_path),
                }
            save_boundary()
            if previous is not None:
                Path(previous["path"]).unlink(missing_ok=True)

        group_offset = progress_after(
            completed[:index],
            0,
            config.project_root,
            config.model_name,
            identity.run_id,
        )
        group.run(
            artifact_before(index),
            prior is not None,
            identity,
            PhasePublication(save_progress, global_step_offset=group_offset),
            revision or {},
            environment,
            previous_commit=completed[index - 1] if index else None,
        )
        native_records, current = strategy.broadcast(
            (completed.records(), state["active"])
        )
        if not writer:
            for record in native_records[len(completed) :]:
                completed.append(PhaseCommit.load(record))
            state["active"] = current
        group.release()
        stop = completed[-1].result["stop_scope"]
        if stop != "end_plan" and len(completed) != index + len(group.phases):
            raise ValueError("native phase group returned before committing its phases")
        return PhaseOutcome(None, stop)

    def finish_group(transition, outcome):
        if transition.phase.index <= last_data_phase < len(completed):
            prepared.release_data()

    next_index = state["active"]["index"] if state["active"] else len(completed)
    start = next(
        (offset for offset, group in enumerate(groups) if group.index == next_index),
        len(groups),
    )
    stopped = bool(completed and completed[-1].result["stop_scope"] == "end_plan")
    if start == len(groups) and next_index != len(phases) and not stopped:
        raise ValueError("learning run checkpoint is between inconsistent phase groups")
    if resuming and state["active"] is None and (start == len(groups) or stopped):
        report = strategy.broadcast(
            completed_run_report(config, checkpoint, identity.run_id, completed)
            if writer
            else None
        )
        if report is not None:
            return _learning_result(config, checkpoint, report)
    completion_reason = run_phases(
        groups,
        run_group,
        finish_group,
        start=start,
        stopped=stopped and state["active"] is None,
        replay=lambda: publish_phase_commits(completed) if writer else None,
    )

    def publish_final():
        from sequifier.rl.checkpoint import publish_policy_model

        final = read_execution_artifact(completed[-1].model_path)
        graph_paths = policy_graph_paths(config)
        publish_policy_model(config, final, graph_paths)
        initialization = (
            completed[0]
            .result["provenance"]
            .get("actor_initialization", {"path": config.initialize_from})
        )
        prepared.close()
        manifest_report = run_report(
            config.outputs,
            checkpoint,
            identity,
            completed,
            resume=resume,
            additional_exports=graph_paths.values(),
            completion_reason=completion_reason,
            source_provenance={
                **manifest.identity,
                **{
                    path: checksum
                    for inputs in state.get("deferred_dependencies", {}).values()
                    for path, checksum in inputs.items()
                },
            },
            initialization=initialization,
            backbone_parent_revision_id=state["backbone_revision"],
            backbone_publication=publish_selected_backbone(
                config,
                final.model_state_dict,
                state["backbone_revision"],
                identity.run_id,
                completed[-1].result,
            ),
            plan_fingerprint=plan_identity,
            resource_accounting=prepared.accounting.snapshot(),
        )
        publish_run_report(manifest_report, config)
        return manifest_report

    manifest_report = on_writer(strategy, publish_final)
    return _learning_result(config, checkpoint, manifest_report)


def published_training_result(config):
    """Return a local view of the result written by a spawned worker group."""
    import json

    from sequifier.artifacts.reports import PhaseResult
    from sequifier.rl.checkpoint import load_policy
    from sequifier.rl.learner import restore_policy_versions
    from sequifier.rl.run_state import ActorRepresentation, PolicyRunResult

    report = json.loads(Path(config.outputs.report_path).read_text())
    checkpoint = report["checkpoint_path"]
    if config.run_mode == "coordinated":
        return _learning_result(config, checkpoint, report)
    actor = load_policy(
        report["model_path"],
        device="cpu",
        interface_name=config.training_plan[-1].model_interface,
    )
    restore_policy_versions(actor, config.training_plan, report["phases"])
    actor.transfer_report = report["initialization"]["transfer"]
    return PolicyRunResult(
        actor.eval(),
        tuple(PhaseResult.model_validate(phase) for phase in report["phases"]),
        ActorRepresentation.from_actor(actor),
        report["model_path"],
        config.outputs.report_path,
        checkpoint,
    )


def _learning_result(config, checkpoint, manifest_report):
    return LearningRunResult(
        config.outputs.model_path,
        str(checkpoint),
        config.outputs.report_path,
        tuple(
            {
                **{key: report[key] for key in ("name", "algorithm", "model_path")},
                "report": report,
            }
            for report in manifest_report["phases"]
        ),
    )
