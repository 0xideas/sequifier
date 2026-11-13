"""Sequential policy coordination over prepared data and validated recovery state."""

from dataclasses import replace
from pathlib import Path

import torch

from sequifier.artifacts.phase import PhaseArtifacts
from sequifier.fingerprints import fingerprint
from sequifier.helpers import configure_determinism
from sequifier.io.atomic import write_json_atomic
from sequifier.rl.checkpoint import load_policy, phase_actor_path, save_policy
from sequifier.rl.preparation import (
    build_actor,
    preflight_environments,
    preflight_optimization,
    prepare_data,
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
    PolicyRunResult,
    clone_weights,
    require_equal,
    weight_fingerprint,
)
from sequifier.runtime.random_state import capture_rng_state, restore_rng_state


def _plan_contract(config):
    values = config.model_dump(mode="json")
    values.pop("resume")
    values.pop("checkpoint_path", None)
    for phase in values["phases"]:
        for role in phase["optimization"].values():
            if role["scheduler"]["name"] == "OneCycleLR":
                # total_steps is an assertion, not a separately configurable budget.
                role["scheduler"]["arguments"].pop("total_steps", None)
    return values


def _initialize_actor(config):
    if config.initialize_from:
        source = load_policy(
            config.initialize_from,
            device=config.device,
            interface_name=config.policy_interface,
            binding=config.policy,
            model=config.model,
        )
        validate_declared_policy(config.policy, source.config, "initialize_from.policy")
        from sequifier.artifacts.model_config import validate_execution_attachment
        from sequifier.artifacts.model_export import model_execution_config

        validate_execution_attachment(
            config.execution_model, model_execution_config(source.execution_config)
        )
        source.execution_config.embedding_layer_names = config.embedding_layer_names
        source.units = dict(config.units)
        return source
    else:
        from sequifier.artifacts.model_config import resolved_config_from_model_config

        execution, _ = resolved_config_from_model_config(
            config.execution_model,
            device=config.device,
            interface_name=config.policy_interface,
        )
        return build_actor(
            config,
            config.policy,
            execution.interface(config.policy_interface).selected_columns_statistics,
        )


def run_policy_training(config: ResolvedPolicyRun) -> PolicyRunResult:
    """Validate the whole run, execute phase-local lifecycles, and export its final actor."""
    if not isinstance(config, ResolvedPolicyRun):
        from sequifier.config.train_config import (
            ResolvedSequifierConfig,
            SequifierConfig,
        )
        from sequifier.config.training_resolution import (
            policy_run_config,
            resolve_training_run,
        )

        if isinstance(config, SequifierConfig):
            config = resolve_training_run(config)
        if not isinstance(config, ResolvedSequifierConfig) or any(
            phase.algorithm == "supervised" for phase in config.training_plan
        ):
            raise TypeError(
                "policy training requires a resolved policy-only training plan"
            )
        config = policy_run_config(config, config.training_plan)
    torch.set_float32_matmul_precision("highest")
    configure_determinism(config.seed, config.enforce_determinism)
    plan = _plan_contract(config)
    inputs = PreparedTrainingInputs.for_run(config).resolve(config)
    inputs.revalidate(inputs.identity, "run.source_identity")
    data_identity = inputs.identity
    validate_output_paths(config, inputs)
    checkpoint = Path(config.checkpoint_path)
    recovery = RunCheckpoint.load(checkpoint, config, plan, data_identity)
    resuming = recovery is not None
    adapter_contract = preflight_environments(config)
    if recovery is not None:
        require_equal(
            adapter_contract, recovery.adapter_contract, "resume.environments"
        )
        representation = recovery.representation
        validate_declared_policy(config.policy, representation.policy, "resume.policy")
        data = prepare_data(config, representation.policy, inputs)
        actor = build_actor(
            config, representation.policy, representation.values["statistics"]
        )
        recovery.validate_actor(actor, config.units)
    else:
        data = prepare_data(config, config.policy, inputs)
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
    references = PolicyReferences(config, recovery, resuming=resuming)
    saved_random = capture_rng_state(actor.device)
    step_budgets = preflight_optimization(config, actor, data)
    restore_rng_state(actor.device, recovery.random if resuming else saved_random)

    if not resuming:
        # Persist the complete fitted representation before the first update.
        recovery.save(checkpoint, actor.device)
    recovery.write_reports()
    while recovery.phase_index < len(config.phases) and not recovery.stopped:
        phase_index = recovery.phase_index
        current_state = recovery.phase_state
        phase = config.phases[phase_index]
        inputs.revalidate(
            data_identity,
            f"phases.{phase.name}.source_identity",
        )
        validate_output_paths(config, inputs)
        incoming_weight_hash = weight_fingerprint(actor.network.state_dict())
        actor.eval()
        reference, reference_identity = references.resolve(actor)

        def commit(state):
            representation.validate_actor(
                actor, config.units, f"phases.{phase.name}.actor"
            )
            recovery.save(checkpoint, actor.device, state)

        if recovery.continuation is not None:
            actor._phase_continuation = recovery.continuation

        context = PhaseContext(
            config,
            phase_index,
            phase,
            actor,
            data[phase_index],
            reference,
            reference_identity,
            current_state,
            commit,
            step_budgets[phase_index],
        )
        actor, result = BUILTIN_ALGORITHMS[phase.algorithm].runtime(context)
        representation.validate_actor(
            actor, config.units, f"phases.{phase.name}.selected_actor"
        )
        actor_path = phase_actor_path(config.phase_directory, phase_index, phase.name)
        result = replace(
            result,
            provenance={
                **result.provenance,
                "config_fingerprint": fingerprint(phase.model_dump(mode="json")),
                "resolved_phase": phase.model_dump(mode="json"),
                "representation_fingerprint": representation.identity,
                "incoming_weight_hash": incoming_weight_hash,
                **PhaseArtifacts(actor_path).describe(),
            },
        )
        recovery.continuation = getattr(actor, "_phase_continuation", None)
        recovery.finish_phase(actor, result)
        recovery.save(checkpoint, actor.device)
        recovery.write_reports()
    report = {
        "format": "sequifier.policy-run-report",
        "version": 1,
        "plan": plan,
        "plan_fingerprint": fingerprint(plan),
        "representation_fingerprint": representation.identity,
        "source_provenance": data_identity,
        "reference_provenance": recovery.reference_contract,
        "initialization": recovery.initialization,
        "resume": config.resume.model_dump(mode="json"),
        "phases": [r.as_dict() for r in recovery.results],
        "model_path": config.model_path,
        "checkpoint_path": config.checkpoint_path,
    }
    save_policy(actor.eval(), config.model_path, metadata=report)
    write_json_atomic(config.report_path, report)
    recovery.save(checkpoint, actor.device)
    return PolicyRunResult(
        actor,
        tuple(recovery.results),
        representation,
        config.model_path,
        config.report_path,
        config.checkpoint_path,
    )
