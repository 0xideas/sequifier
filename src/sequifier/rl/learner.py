"""Native policy phases over prepared data and validated recovery state."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from sequifier.artifacts.dependencies import add_initialization
from sequifier.artifacts.phase import PhaseArtifacts, publish_phase_commits
from sequifier.artifacts.reports import publish_run_report, run_report
from sequifier.fingerprints import fingerprint
from sequifier.helpers import configure_determinism, configure_logger
from sequifier.io.output_paths import validate_training_destinations
from sequifier.rl.checkpoint import save_policy, validate_policy_weights
from sequifier.rl.preparation import (
    PolicyGroupPreparation,
    build_actor,
    preflight_optimization,
    resolve_normalization,
    validate_declared_policy,
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
    PolicyPhaseResult,
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
    from sequifier.artifacts.run_checkpoint import (
        phase_contract,
        training_plan_identity,
    )

    return {
        "resolved_run": training_plan_identity(config.parent),
        "seed": config.parent.seed,
        "phase_offset": config.phase_offset,
        "phases": [phase_contract(phase) for phase in config.phases],
        "initialize_from": config.initialize_from,
    }


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
    decoder = actor.network.resolve_interface(name).require_decoder()
    history_ids = history_parameter_ids(actor.network, name)
    if any(id(parameter) in history_ids for parameter in decoder.parameters()):
        raise ValueError(f"{name}: history_only requires an untied action decoder")
    checksum = file_checksum(config.initialize_from)
    random = capture_rng_state(actor.device)
    try:
        source, execution, _ = load_execution_model(
            config.initialize_from, device=config.parent.device, interface_name=name
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
        validate_execution_attachment(
            artifact.model_config.values, config.parent.execution
        )
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
        prepared = prepared or PreparedTrainingInputs()
        with prepared.run_scope():
            return _execute_policy(config, prepared)

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
    preparation: PolicyGroupPreparation,
):
    """Run a group of policy phases with a shared actor and optimizer lifecycle."""
    _execute_policy(
        config,
        prepared,
        identity=identity,
        on_checkpoint=on_checkpoint,
        preparation=preparation,
    )


def _execute_policy(
    config,
    prepared,
    *,
    identity=None,
    on_checkpoint=None,
    preparation=None,
):
    group = preparation is not None
    torch.set_float32_matmul_precision("highest")
    configure_determinism(
        config.parent.seed, config.parent.runtime_settings.enforce_determinism
    )
    for index, phase in enumerate(config.phases):
        validate_continuation(
            config.phases[index - 1] if index else None, phase, resolved=True
        )
    phase_identities = tuple(
        PolicyPhaseIdentity.resolve(index, config.parent.seed)
        for index, _ in enumerate(config.phases, config.phase_offset)
    )
    plan = _plan_contract(config)
    checkpoint = Path(config.checkpoint_path)
    recovery = RunCheckpoint.load(
        checkpoint,
        config,
        plan,
        run_id=identity.run_id if identity is not None else None,
    )
    resuming = recovery is not None
    binding = config.policy
    if recovery is not None:
        binding = recovery.representation.policy
        validate_declared_policy(config.policy, binding, "resume.policy")
    if preparation is None:
        preparation = PolicyGroupPreparation.build(
            config,
            prepared,
            binding,
            saved=recovery.data_identity if recovery is not None else None,
            committed_reference_count=(
                recovery.phase_index + int(recovery.phase_state is not None)
                if recovery is not None
                else 0
            ),
        )
    else:
        # The selected handoff only exists once the preceding group has committed.
        add_initialization(preparation.dependencies, config)
        validate_training_destinations(
            config,
            config.checkpoint_path,
            preparation.dependencies.protected_paths,
            phase_offset=config.phase_offset,
            group=True,
        )
    dependencies = preparation.dependencies
    data_identity = dependencies.identity
    adapter_contract = preparation.adapters
    data = preparation.data
    if binding != preparation.binding:
        data = preparation.prepare_data(config, prepared, preparation.inputs, binding)
    configure_logger(config.project_root, config.model_name, 0)
    if recovery is not None:
        recovery.validate_dependencies(config, data_identity)
        require_equal(
            adapter_contract, recovery.adapter_contract, "resume.environments"
        )
        representation = recovery.representation
        actor = build_actor(config, binding)
        recovery.validate_actor(actor, config.policy.units)
    else:
        actor = _initialize_actor(config)
        representation = ActorRepresentation.from_actor(actor, config.policy.units)
    normalization = config.parent.normalization_by_interface.get(
        config.policy_interface
    )
    if normalization is not None:
        expected_normalization = resolve_normalization(actor.config, normalization)
        require_equal(
            expected_normalization,
            {
                key: value
                for key, value in representation.values["statistics"].items()
                if key in expected_normalization
            },
            "run.normalization",
        )
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
                "transfer": actor.transfer_report,
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

    def save_checkpoint(state=None, *, commit=None):
        recovery.save(checkpoint, actor.device, state)
        if on_checkpoint is not None:
            on_checkpoint(checkpoint, commit)

    if not resuming:
        # Persist the complete fitted representation before the first update.
        save_checkpoint()

    def run_phase(transition):
        position, phase = transition.index, transition.phase
        phase_identity = phase_identities[position]
        phase_index = phase_identity.index
        current_state = recovery.phase_state
        dependencies.revalidate()
        incoming_weight_hash = weight_fingerprint(actor.network.state_dict())
        initialization = {}
        if phase.uses_actor_history:
            from sequifier.model.history_transfer import history_components

            initialization["history_initialization"] = {
                "source": "phase_start_actor",
                "source_weight_hash": incoming_weight_hash,
                "transferred_components": list(
                    history_components(actor.network, actor.interface_name)
                ),
                "fresh_components": optimizations[position].fresh_history_components,
            }
        if phase_index == 0 and config.actor_initialization == "history_only":
            initialization["actor_initialization"] = recovery.initialization
        actor.eval()
        reference, reference_identity = references.resolve(actor, phase_index)

        def commit(state):
            representation.validate_actor(
                actor, config.policy.units, f"phases.{phase.name}.actor"
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
                adapter_contract,
                metrics=metrics,
                continuation=recovery.continuation
                if transition.continues_optimizer
                else None,
                preserve_optimizer=transition.preserves_optimizer,
            )
            execution = BUILTIN_ALGORITHMS[phase.algorithm].runtime(context)
            telemetry_position = telemetry.commit()
        representation.validate_actor(
            execution.actor, config.policy.units, f"phases.{phase.name}.selected_actor"
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
        commit = recovery.finish_phase(execution, position, checkpoint)
        save_checkpoint(commit=commit)
        return commit

    def publish(record):
        record.publish()
        if record.result["index"] == config.phase_offset + last_data_phase:
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
        replay=None if group else lambda: publish_phase_commits(recovery.completed),
        publish=None if group else publish,
        optimizer_continues=lambda phase: phase.optimizer_state == "continue",
    )
    results = ()
    if not group:
        reports = [record.report_result() for record in recovery.completed]
        results = tuple(PolicyPhaseResult.from_report(report) for report in reports)
        data.clear()
        prepared.close()
        report = run_report(
            model_path=config.model_path,
            checkpoint_path=checkpoint,
            phases=reports,
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
