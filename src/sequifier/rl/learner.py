"""Native policy phases over prepared data and validated recovery state."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from sequifier.artifacts.dependencies import policy_dependencies
from sequifier.artifacts.phase import PhaseArtifacts
from sequifier.artifacts.reports import phase_report, publish_run_report, run_report
from sequifier.fingerprints import fingerprint
from sequifier.helpers import configure_determinism, configure_logger
from sequifier.rl.checkpoint import save_policy, validate_policy_weights
from sequifier.rl.preparation import (
    build_actor,
    preflight_environments,
    preflight_optimization,
    resolve_normalization,
    validate_data,
    validate_declared_policy,
    validate_output_paths,
)
from sequifier.rl.prepared_inputs import PreparedTrainingInputs
from sequifier.rl.references import PolicyReferences
from sequifier.rl.registry import BUILTIN_ALGORITHMS
from sequifier.rl.run_checkpoint import RunCheckpoint
from sequifier.rl.run_config import ResolvedPolicyRun
from sequifier.rl.run_state import (
    ActorRepresentation,
    PhaseContext,
    PolicyPhaseIdentity,
    PolicyRunResult,
    clone_weights,
    require_equal,
    weight_fingerprint,
)
from sequifier.rl.telemetry import TelemetryJournal
from sequifier.runtime.plan import (
    PhaseOutcome,
    RunIdentity,
    run_phases,
    validate_continuation,
)
from sequifier.runtime.random_state import capture_rng_state, restore_rng_state

if TYPE_CHECKING:
    from sequifier.config.train_config import ResolvedSequifierConfig, SequifierConfig
    from sequifier.training.coordinator import LearningRunResult


def _plan_contract(config):
    from pydantic_core import to_jsonable_python

    from sequifier.artifacts.model_export import model_execution_config

    values = config.model_dump(mode="json")
    if config.phase_offset:
        values["phase_offset"] = config.phase_offset
    values["execution_model"] = to_jsonable_python(
        model_execution_config(config.execution)
    )
    values["model"] = config.execution.model.model_dump(mode="json")
    values.pop("resume")
    values.pop("checkpoint_path", None)
    if config.actor_initialization == "attach":
        values.pop("actor_initialization")
    for phase in values["phases"]:
        for role in phase["optimization"].values():
            if role["scheduler"]["name"] == "OneCycleLR":
                # total_steps is an assertion, not a separately configurable budget.
                role["scheduler"]["arguments"].pop("total_steps", None)
    return values


def _phase_report(result, checkpoint):
    return phase_report(
        model_path=result.provenance["actor_path"],
        checkpoint_path=checkpoint,
        **result.as_dict(),
    )


def _publish_phase_report(record, checkpoint):
    result = record.report_result()
    artifacts = PhaseArtifacts(Path(result.provenance["actor_path"]))
    record.commit.publish(artifacts, _phase_report(result, checkpoint))


def _initialize_history_actor(config):
    from sequifier.artifacts.model_artifact import load_execution_model
    from sequifier.model.history_transfer import (
        copy_history_weights,
        history_parameter_ids,
        validate_history_transfer,
    )
    from sequifier.rl.models import validate_independent_networks
    from sequifier.trajectory.sources import file_checksum

    if not config.initialize_from:
        raise ValueError("history_only requires initialize_from")
    actor = build_actor(config)
    name = actor.interface_name
    decoder = actor.network.resolve_interface(name).decoder
    history_ids = history_parameter_ids(actor.network, name)
    if any(id(parameter) in history_ids for parameter in decoder.parameters()):
        raise ValueError(f"{name}: history_only requires an untied action decoder")
    checksum = file_checksum(config.initialize_from)
    random = capture_rng_state(actor.device)
    try:
        source, execution, _ = load_execution_model(
            config.initialize_from, device=config.device, interface_name=name
        )
    finally:
        restore_rng_state(actor.device, random)
    require_equal(
        checksum, file_checksum(config.initialize_from), "initialize_from.sha256"
    )
    validate_history_transfer(execution, actor.execution_config, name)
    validate_independent_networks(source, actor.network)
    components = copy_history_weights(source, actor.network, name)
    validate_policy_weights(actor)
    actor.transfer_report = {
        "mode": "history_only",
        "interface": name,
        "source_sha256": checksum,
        "source_preprocessing": "preserved",
        "transferred_components": components,
        "fresh_components": [f"interfaces.{name}.decoder"],
    }
    return actor


def _initialize_actor(config):
    if config.actor_initialization == "history_only":
        return _initialize_history_actor(config)
    if config.initialize_from:
        from sequifier.artifacts.model_artifact import read_execution_artifact
        from sequifier.artifacts.model_config import validate_execution_attachment
        from sequifier.rl.attachment import attachment_report

        artifact = read_execution_artifact(config.initialize_from)
        validate_execution_attachment(artifact.model_config.values, config.execution)
        actor = build_actor(config, artifact=artifact)
        actor.transfer_report = attachment_report(actor, artifact.model_state_dict)
        return actor
    return build_actor(config)


def run_policy_training(
    config: ResolvedPolicyRun | ResolvedSequifierConfig | SequifierConfig,
    *,
    prepared: PreparedTrainingInputs | None = None,
) -> PolicyRunResult | LearningRunResult:
    """Validate the whole run, execute phase-local lifecycles, and export its final actor."""
    if isinstance(config, ResolvedPolicyRun):
        return _run_policy(config, prepared or PreparedTrainingInputs())

    from sequifier.config.train_config import ResolvedSequifierConfig, SequifierConfig
    from sequifier.config.training_resolution import resolve_training_run
    from sequifier.training.coordinator import run_policy_plan

    if isinstance(config, SequifierConfig):
        config = resolve_training_run(config)
    if not isinstance(config, ResolvedSequifierConfig) or any(
        phase.algorithm == "supervised" for phase in config.training_plan
    ):
        raise TypeError("policy training requires a resolved policy-only training plan")
    return run_policy_plan(config)


def run_policy_group(
    config: ResolvedPolicyRun,
    identity: RunIdentity,
    on_checkpoint,
    *,
    prepared: PreparedTrainingInputs,
):
    """Run a group of policy phases with a shared actor and optimizer lifecycle."""
    _run_policy(
        config, prepared, group=True, identity=identity, on_checkpoint=on_checkpoint
    )


def _run_policy(config, prepared, *, group=False, identity=None, on_checkpoint=None):
    with prepared.run_scope():
        return _execute_policy(config, prepared, group, identity, on_checkpoint)


def _execute_policy(config, prepared, group, identity, on_checkpoint):
    torch.set_float32_matmul_precision("highest")
    configure_determinism(config.seed, config.enforce_determinism)
    for index, phase in enumerate(config.phases):
        validate_continuation(
            config.phases[index - 1] if index else None, phase, resolved=True
        )
    phase_identities = tuple(
        PolicyPhaseIdentity.resolve(index, config.seed)
        for index, _ in enumerate(config.phases, config.phase_offset)
    )
    plan = _plan_contract(config)
    inputs = prepared.resolve(config)
    inputs.revalidate(inputs.identity, "run.source_identity")
    checkpoint = Path(config.checkpoint_path)
    recovery = RunCheckpoint.load(
        checkpoint,
        config,
        plan,
        run_id=identity.run_id if identity is not None else None,
    )
    resuming = recovery is not None
    dependencies = policy_dependencies(
        config,
        prepared,
        inputs,
        saved=recovery.data_identity if recovery is not None else None,
        committed_reference_count=(
            recovery.phase_index + int(recovery.phase_state is not None)
            if recovery is not None
            else 0
        ),
    )
    data_identity = dependencies.identity
    if recovery is not None:
        recovery.validate_dependencies(config, data_identity)
    validate_output_paths(config, dependencies)
    configure_logger(config.project_root, config.model_name, 0)
    adapter_contract = preflight_environments(config)
    if recovery is not None:
        require_equal(
            adapter_contract, recovery.adapter_contract, "resume.environments"
        )
        representation = recovery.representation
        validate_declared_policy(config.policy, representation.policy, "resume.policy")
        data = prepared.prepare(config, representation.policy, inputs)
        actor = build_actor(config, representation.policy)
        recovery.validate_actor(actor, config.units)
    else:
        data = prepared.prepare(config, config.policy, inputs)
        actor = _initialize_actor(config)
        representation = ActorRepresentation.from_actor(actor, config.units)
    if config.normalization is not None:
        expected_normalization = resolve_normalization(
            actor.config, config.normalization
        )
        require_equal(
            expected_normalization,
            {
                key: value
                for key, value in representation.values["statistics"].items()
                if key in expected_normalization
            },
            "run.normalization",
        )
    validate_data(data, actor)
    prepared.measure()
    data_contract = [item.contract if item is not None else None for item in data]
    if recovery is not None:
        require_equal(data_contract, recovery.data_contract, "resume.partitions")
    if recovery is None:
        recovery = RunCheckpoint(
            plan,
            representation,
            data_identity,
            data_contract,
            adapter_contract,
            {
                "path": config.initialize_from,
                "transfer": getattr(actor, "transfer_report", None),
                "weight_hash": weight_fingerprint(actor.network.state_dict()),
            },
            clone_weights(actor.network),
            capture_rng_state(actor.device),
        )
    from sequifier.rl.metrics import PolicyMetrics

    identity = identity or RunIdentity.start(recovery.run_id)
    recovery.run_id = identity.run_id
    metrics = PolicyMetrics(config, identity)
    references = PolicyReferences(config, recovery, resuming=resuming)
    saved_random = capture_rng_state(actor.device)
    optimizations = preflight_optimization(config, actor, data)
    restore_rng_state(actor.device, recovery.random if resuming else saved_random)

    def save_checkpoint(state=None):
        recovery.save(checkpoint, actor.device, state)
        if on_checkpoint is not None:
            on_checkpoint(
                checkpoint, [record.as_dict() for record in recovery.completed]
            )

    if not resuming:
        # Persist the complete fitted representation before the first update.
        save_checkpoint()

    def replay():
        for record in recovery.completed:
            _publish_phase_report(record, checkpoint)

    def run_phase(transition):
        position, phase = transition.index, transition.phase
        phase_identity = phase_identities[position]
        phase_index = phase_identity.index
        current_state = recovery.phase_state
        dependencies.revalidate()
        incoming_weight_hash = weight_fingerprint(actor.network.state_dict())
        initialization = {}
        if getattr(phase, "history_initialization", None) == "phase_start_actor":
            from sequifier.model.history_transfer import history_components

            initialization["history_initialization"] = {
                "source": "phase_start_actor",
                "source_weight_hash": incoming_weight_hash,
                "transferred_components": list(
                    history_components(actor.network, actor.interface_name)
                ),
                "fresh_components": {"value": ["head"]},
            }
            if phase.algorithm == "iql":
                initialization["history_initialization"]["fresh_components"].update(
                    {name: ["action_embedding", "head"] for name in ("q1", "q2")}
                )
        if phase_index == 0 and config.actor_initialization == "history_only":
            initialization["actor_initialization"] = recovery.initialization
        actor.eval()
        reference, reference_identity = references.resolve(actor, phase_index)

        def commit(state):
            representation.validate_actor(
                actor, config.units, f"phases.{phase.name}.actor"
            )
            save_checkpoint({**state, "telemetry_path": str(artifacts.telemetry)})

        artifacts = PhaseArtifacts.for_phase(
            config.phase_directory, phase_index, phase.name
        )
        telemetry_position = (
            current_state["telemetry_position"] if current_state else None
        )
        with TelemetryJournal.open(
            artifacts.telemetry, telemetry_position
        ) as telemetry:
            context = PhaseContext(
                config,
                phase_identity,
                phase,
                actor,
                data[position] if data else None,
                reference,
                reference_identity,
                current_state,
                commit,
                optimizations[position],
                telemetry,
                metrics=metrics,
                continuation=recovery.continuation
                if transition.continues_optimizer
                else None,
                preserve_optimizer=transition.preserves_optimizer,
            )
            execution = BUILTIN_ALGORITHMS[phase.algorithm].runtime(context)
            telemetry_position = telemetry.commit()
        representation.validate_actor(
            execution.actor, config.units, f"phases.{phase.name}.selected_actor"
        )
        result = execution.result
        result = replace(
            result,
            provenance={
                **result.provenance,
                "config_fingerprint": fingerprint(phase.model_dump(mode="json")),
                "resolved_phase": phase.model_dump(mode="json"),
                "representation_fingerprint": representation.identity,
                "incoming_weight_hash": incoming_weight_hash,
                **initialization,
                **artifacts.describe(),
                "resource_accounting": prepared.accounting.snapshot(),
                "source_provenance": data_identity,
            },
        )
        return PhaseOutcome(
            (replace(execution, result=result), telemetry_position), result.stop_scope
        )

    def commit_phase(transition, outcome):
        nonlocal actor
        execution, position = outcome.result
        actor = execution.actor
        recovery.finish_phase(execution, position)
        save_checkpoint()
        return recovery.completed[-1]

    def publish(record):
        _publish_phase_report(record, checkpoint)
        if not group and record.result.index == config.phase_offset + last_data_phase:
            release_phase_data()

    last_data_phase = max(
        (index for index, part in enumerate(data) if part is not None), default=-1
    )

    def release_phase_data():
        data.clear()
        prepared.release_data()

    if not group and last_data_phase < recovery.phase_index:
        release_phase_data()
    completion_reason = run_phases(
        config.phases,
        run_phase,
        commit_phase,
        start=recovery.phase_index,
        stopped=recovery.stopped,
        replay=replay,
        publish=publish,
        optimizer_continues=lambda phase: phase.optimizer_state == "continue",
    )
    results = ()
    if not group:
        results = tuple(record.report_result() for record in recovery.completed)
        data.clear()
        prepared.close()
        report = run_report(
            model_path=config.model_path,
            checkpoint_path=checkpoint,
            phases=[_phase_report(result, checkpoint) for result in results],
            run_id=recovery.run_id,
            session_id=identity.session_id,
            status="complete",
            completion_reason=completion_reason,
            exports=[config.model_path],
            plan=plan,
            plan_fingerprint=fingerprint(plan),
            representation_fingerprint=representation.identity,
            source_provenance=data_identity,
            reference_provenance=recovery.reference_contract,
            initialization=recovery.initialization,
            resume=config.resume.model_dump(mode="json"),
            resource_accounting=prepared.accounting.snapshot(),
        )
        save_policy(actor.eval(), config.model_path, metadata=report)
        publish_run_report(report, config.report_path)
    return PolicyRunResult(
        actor,
        results,
        representation,
        config.model_path,
        config.report_path,
        config.checkpoint_path,
    )
