"""Native policy phases over prepared data and validated recovery state."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from socket import gethostname
from typing import TYPE_CHECKING

import torch

from sequifier.artifacts.backbone_repository import (
    load_revision,
    publish_selected_backbone,
    select_revision,
)
from sequifier.artifacts.dependencies import add_backbone_revision, add_initialization
from sequifier.artifacts.phase import CommittedPhases, PhaseArtifacts, PhasePublication
from sequifier.artifacts.reports import (
    PhaseResult,
    completed_run_report,
    publish_run_report,
    run_report,
)
from sequifier.artifacts.state_dict import weight_fingerprint
from sequifier.fingerprints import fingerprint
from sequifier.helpers import configure_determinism, configure_logger
from sequifier.io.output_paths import policy_graph_paths, validate_training_destinations
from sequifier.rl.checkpoint import (
    policy_model_artifact,
    publish_policy_model,
    validate_policy_weights,
)
from sequifier.rl.compilation import compiled_policies
from sequifier.rl.preparation import (
    PolicyGroupPreparation,
    build_actor,
    preflight_optimization,
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
    require_equal,
)
from sequifier.rl.telemetry import TelemetryJournal
from sequifier.runtime.plan import (
    PhaseOutcome,
    RunIdentity,
    run_phases,
    validate_continuation,
)
from sequifier.runtime.precision import PrecisionPolicy
from sequifier.runtime.random_state import capture_rng_state, restore_rng_state
from sequifier.training.metrics import progress_after

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


def restore_policy_versions(actor, phases, results):
    for phase, result in zip(phases, results):
        clocks = result["clocks"]
        if phase.algorithm == "ppo":
            width = phase.collection.env_steps_per_iteration
            actor.ppo_policy_version = (clocks["env_steps"] + width - 1) // width
        elif phase.algorithm == "grpo":
            width = phase.collection.groups_per_iteration
            actor.grpo_policy_version = (clocks["groups"] + width - 1) // width


def _initialize_history_actor(config):
    from sequifier.artifacts.model_artifact import load_execution_model
    from sequifier.fingerprints import file_checksum
    from sequifier.model.history_transfer import (
        copy_history_weights,
        history_parameter_ids,
        validate_history_transfer,
    )
    from sequifier.rl.models import validate_independent_networks

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
            config.initialize_from, device=str(actor.device), interface_name=name
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
    config: ResolvedSequifierConfig | SequifierConfig,
) -> PolicyRunResult | LearningRunResult | None:
    """Resolve and launch a policy-only training plan."""
    from sequifier.config.train_config import ResolvedSequifierConfig, SequifierConfig
    from sequifier.config.training_resolution import resolve_training_run
    from sequifier.training.coordinator import run_policy_plan

    if isinstance(config, SequifierConfig):
        config = resolve_training_run(config)
    if not isinstance(config, ResolvedSequifierConfig) or any(
        phase.algorithm == "supervised"
        for phase in (
            config._rollout_plan.ordered_phases
            if config._rollout_plan
            else config.training_plan
        )
    ):
        raise TypeError("policy training requires a resolved policy-only training plan")
    return run_policy_plan(config)


def execute_policy(
    config: ResolvedPolicyRun,
    prepared: PreparedTrainingInputs,
    *,
    identity=None,
    publication: PhasePublication = PhasePublication(),
    preparation=None,
    backbone_revision=None,
    environment=None,
):
    from sequifier.training.distributed_strategy import (
        LocalStrategy,
        on_writer,
        strategy_for,
    )

    device = environment.device if environment else torch.device(config.parent.device)
    settings = config.parent.run.execution
    precision = PrecisionPolicy.from_settings(settings, device)
    strategy = (
        strategy_for(
            environment,
            settings.distribution_settings,
            find_unused=True,
            mixed_precision_dtype=precision.autocast_dtype,
            preserve_io_dtype=True,
        )
        if environment
        else LocalStrategy(device=device)
    )
    writer = strategy.rank == 0
    configure_determinism(
        config.parent.seed, config.parent.run.reproducibility.enforce_determinism
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
    terminal = recovery is not None and (
        recovery.phase_index == len(config.phases) or recovery.stopped
    )
    if publication.standalone and not config.initialize_from:
        if recovery is None:
            backbone_revision = on_writer(
                strategy,
                lambda: select_revision(
                    config.parent.execution.backbone, config.project_root
                ),
            )
        elif pinned := recovery.initialization.get("backbone_parent_revision_id"):
            backbone_revision = select_revision(
                config.parent.execution.backbone,
                config.project_root,
                revision_id=pinned,
            )
    parent_revision_id = backbone_revision["revision_id"] if backbone_revision else None
    if preparation is None:
        preparation = PolicyGroupPreparation.build(
            config,
            prepared,
            backbone_revision_id=parent_revision_id,
            saved=recovery.data_identity if recovery is not None else None,
            known_adapters=recovery.adapter_contract if recovery and terminal else None,
            committed_reference_count=(
                recovery.phase_index + int(recovery.phase_state is not None)
                if recovery is not None
                else 0
            ),
        )
    else:
        # The selected handoff only exists once the preceding group has committed.
        add_initialization(preparation.dependencies, config)
        add_backbone_revision(
            preparation.dependencies, config.parent, parent_revision_id
        )
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
    identity = strategy.broadcast(
        identity or RunIdentity.start(recovery.run_id if recovery else None)
    )
    if recovery is not None:
        recovery.validate_dependencies(config, data_identity)
        require_equal(
            adapter_contract, recovery.adapter_contract, "resume.environments"
        )
        representation = recovery.representation
        actor = build_actor(config, initialize=False)
        recovery.validate_actor(actor)
        actor.transfer_report = recovery.initialization["transfer"]
        restore_policy_versions(
            actor, config.phases, [commit.result for commit in recovery.completed]
        )
    else:
        actor = _initialize_actor(config)
        if backbone_revision and not config.initialize_from:
            load_revision(actor.backbone, backbone_revision)
            validate_policy_weights(actor)
        representation = ActorRepresentation.from_actor(actor)
    normalization = config.parent.normalization_by_interface.get(
        config.policy.interface_name
    )
    if normalization is not None:
        actor.interface.validate_normalization(
            normalization,
            required_columns=actor.config.observation_columns
            + actor.config.static_columns,
        )
    prepared.measure()
    data_contract = [item.contract if item is not None else None for item in data]
    if recovery is not None:
        require_equal(data_contract, recovery.data_contract, "resume.partitions")
        if publication.standalone and terminal:
            report = strategy.broadcast(
                completed_run_report(
                    config, checkpoint, recovery.run_id, recovery.completed
                )
                if writer
                else None
            )
            if report is not None:
                return PolicyRunResult(
                    actor.eval(),
                    tuple(
                        PhaseResult.model_validate(phase) for phase in report["phases"]
                    ),
                    representation,
                    config.outputs.model_path,
                    config.outputs.report_path,
                    config.checkpoint_path,
                )
    logger = configure_logger(
        config.project_root,
        config.model_name,
        strategy.rank,
        rank_specific=strategy.world_size > 1,
    )
    logger.info(f"Starting training for {config.model_name}, run {identity.run_id}")
    if recovery is None:
        incoming_actor = LocalStrategy(device=device).capture_model_state(actor.network)
        recovery = RunCheckpoint(
            plan,
            representation,
            data_identity,
            data_contract,
            adapter_contract,
            {
                "path": config.initialize_from,
                "transfer": actor.transfer_report,
                "backbone_parent_revision_id": parent_revision_id,
                "weight_hash": weight_fingerprint(incoming_actor),
            },
            incoming_actor,
            capture_rng_state(strategy.device),
            completed=CommittedPhases(config.phases, config.phase_offset),
            run_id=identity.run_id,
        )
    from sequifier.rl.metrics import PolicyMetrics

    metrics = PolicyMetrics(config, identity) if writer else None
    if resuming and metrics is not None:
        metrics.writer.restore_positions(identity.run_id, recovery.metric_positions)
    references = PolicyReferences(config, recovery, resuming=resuming)
    saved_random = capture_rng_state(strategy.device)
    optimizations = preflight_optimization(config, actor, data)
    random = recovery.random if resuming else saved_random
    if "ranks" in random:
        if len(random["ranks"]) != strategy.world_size:
            raise ValueError("Policy recovery requires its original world size")
        random = random["ranks"][strategy.rank]
    elif resuming and strategy.world_size != 1:
        raise ValueError("Single-process policy recovery requires world size one")
    restore_rng_state(strategy.device, random)

    def save_checkpoint(state=None, *, commit=None):
        random = capture_rng_state(strategy.device)
        if strategy.world_size > 1:
            random = {"ranks": strategy.gather_for_writer(random)}

        def publish():
            assert metrics is not None
            recovery.metric_positions = metrics.writer.checkpoint_positions()
            recovery.save(checkpoint, strategy.device, state, random=random)
            publication.after_checkpoint(checkpoint, commit)

        on_writer(strategy, publish)

    if not resuming:
        # Persist the complete fitted representation before the first update.
        save_checkpoint()

    def run_phase(transition):
        position, phase = transition.index, transition.phase
        phase_identity = phase_identities[position]
        phase_index = phase_identity.index
        logger.info(f"Starting phase {phase.name} ({phase.algorithm})")
        current_state = recovery.phase_state
        step_offset = progress_after(
            recovery.completed,
            publication.global_step_offset,
            config.project_root,
            config.model_name,
            identity.run_id,
        )
        if current_state:
            if "global_step_offset" in current_state:
                step_offset = current_state["global_step_offset"]
            else:
                # Earlier metric tables used the sum of this phase's role updates.
                legacy_step = sum(
                    role["steps"] for role in current_state["optimization"].values()
                )
                step_offset = max(
                    step_offset, legacy_step - current_state["transaction"]
                )

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
            representation.validate_actor(actor, f"phases.{phase.name}.actor")
            save_checkpoint(
                {
                    **state,
                    "telemetry_path": str(artifacts.telemetry),
                    "global_step_offset": step_offset,
                }
            )

        artifacts = PhaseArtifacts.for_phase(
            config.outputs.phase_directory, phase_index, phase.name
        )
        telemetry_position = (
            current_state["telemetry_position"] if current_state else None
        )
        with TelemetryJournal.open(
            artifacts.telemetry, telemetry_position, writer=writer
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
                strategy=strategy,
                precision=precision,
                global_step_offset=step_offset,
                continuation=recovery.continuation
                if transition.continues_optimizer
                else None,
                preserve_optimizer=transition.preserves_optimizer,
            )
            with compiled_policies(
                config.parent.run.execution.policy_compile_mode, actor, reference
            ):
                execution = BUILTIN_ALGORITHMS[phase.algorithm].runtime(context)
            telemetry_position = telemetry.commit()
        representation.validate_actor(
            execution.actor, f"phases.{phase.name}.selected_actor"
        )
        result = execution.result
        distribution = {}
        if strategy.world_size > 1:
            distribution = {
                "distributed_execution": {
                    "strategy": config.parent.run.execution.distribution_settings.strategy,
                    "world_size": strategy.world_size,
                    "ranks": strategy.gather_for_writer(
                        {
                            "host": gethostname(),
                            "rank": strategy.rank,
                            "local_rank": strategy.local_rank,
                            "device": str(strategy.device),
                            "working_storage": prepared.accounting.snapshot(),
                            "process_peak_cuda_bytes": torch.cuda.max_memory_allocated(
                                strategy.device
                            )
                            if strategy.device.type == "cuda"
                            else None,
                        }
                    ),
                }
            }
        result = result.model_copy(
            update={
                "provenance": {
                    **result.provenance,
                    **distribution,
                    "config_fingerprint": fingerprint(phase.model_dump(mode="json")),
                    "resolved_phase": phase.model_dump(mode="json"),
                    "representation_fingerprint": representation.identity,
                    "incoming_weight_hash": incoming_weight_hash,
                    **initialization,
                    **artifacts.describe(),
                    "resource_accounting": prepared.accounting.snapshot(),
                    "source_provenance": data_identity,
                }
            }
        )
        return PhaseOutcome(
            (replace(execution, result=result), telemetry_position), result.stop_scope
        )

    def commit_phase(transition, outcome):
        nonlocal actor
        execution, position = outcome.result
        actor = execution.actor
        commit = on_writer(
            strategy, lambda: recovery.finish_phase(execution, position, checkpoint)
        )
        save_checkpoint(commit=commit)
        if not writer:
            # Read the durable, consolidated generation from shared storage.
            from sequifier.artifacts.checkpoint_envelope import recovery_payload

            payload = recovery_payload(
                torch.load(checkpoint, map_location="cpu", weights_only=False), "policy"
            )
            recovery.completed.append(commit)
            recovery.incoming_actor = payload["incoming_actor"]
            recovery.continuation = payload["continuation"]
            recovery.stopped = payload["stopped"]
            recovery.phase_state = None
            actor.network.load_state_dict(recovery.incoming_actor)
        if publication.standalone and transition.index == last_data_phase:
            release_phase_data()

    last_data_phase = max(
        (index for index, part in enumerate(data) if part is not None), default=-1
    )

    def release_phase_data():
        data.clear()
        prepared.release_data()

    if publication.standalone and last_data_phase < recovery.phase_index:
        release_phase_data()
    completion_reason = run_phases(
        config.phases,
        run_phase,
        commit_phase,
        start=recovery.phase_index,
        stopped=recovery.stopped,
        replay=lambda: (
            publication.replay_native(recovery.completed) if writer else None
        ),
        optimizer_continues=lambda phase: phase.optimizer_state == "continue",
    )

    def publish_final():
        data.clear()
        prepared.close()
        graph_paths = policy_graph_paths(config.parent)
        report = run_report(
            config.outputs,
            checkpoint,
            identity,
            recovery.completed,
            resume=config.resume,
            additional_exports=graph_paths.values(),
            completion_reason=completion_reason,
            plan=plan,
            plan_fingerprint=fingerprint(plan),
            representation_fingerprint=representation.identity,
            source_provenance=data_identity,
            reference_provenance=recovery.reference_contract,
            initialization=recovery.initialization,
            backbone_parent_revision_id=parent_revision_id,
            backbone_publication=publish_selected_backbone(
                config.parent,
                recovery.incoming_actor,
                parent_revision_id,
                identity.run_id,
                recovery.completed[-1].result,
            ),
            resource_accounting=prepared.accounting.snapshot(),
        )
        results = tuple(PhaseResult.model_validate(phase) for phase in report["phases"])
        artifact = policy_model_artifact(actor.eval(), metadata=report)
        publish_policy_model(config.parent, artifact, graph_paths)
        publish_run_report(report, config)
        return results

    results = on_writer(strategy, publish_final) if publication.standalone else ()
    logger.info(f"Training complete ({completion_reason})")
    return PolicyRunResult(
        actor,
        results,
        representation,
        config.outputs.model_path,
        config.outputs.report_path,
        config.checkpoint_path,
    )
