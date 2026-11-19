"""Read-only policy run preparation and whole-plan preflight validation."""

from dataclasses import replace
from pathlib import Path

import torch

from sequifier.io.output_paths import (
    require_directory_destinations,
    require_file_destinations,
    unique_paths,
    validate_path_conflicts,
)
from sequifier.model.factory import build_transformer_network, validate_hash_contracts
from sequifier.rl.checkpoint import phase_actor_path, validate_policy_weights
from sequifier.rl.environment import (
    adapter_identity,
    load_online_adapter,
    require_exact_snapshot,
    require_group_start,
)
from sequifier.rl.policy import Policy
from sequifier.rl.ppo_evaluation import require_evaluation_timer
from sequifier.rl.prepared_inputs import PreparedTrainingInputs, RunInputs  # noqa: F401
from sequifier.rl.run_state import require_equal
from sequifier.runtime.random_state import capture_rng_state, restore_rng_state


def validate_output_paths(config, dependencies):
    """Keep every output and archive separate from inputs and other outputs."""
    protected = dependencies.protected_paths
    outputs = [
        Path(config.model_path),
        Path(config.report_path),
        Path(config.checkpoint_path),
    ]
    for i, phase in enumerate(config.phases, config.phase_offset):
        actor = phase_actor_path(config.phase_directory, i, phase.name)
        outputs += [actor, actor.with_name("report.json")]
    message = "policy outputs and phase archives must have distinct paths"
    outputs = unique_paths(outputs, message)
    archives = unique_paths(
        [
            Path(archive).resolve()
            for phase in config.phases
            if (archive := phase.rollout_archive_directory)
        ],
        message,
    )
    require_file_destinations(outputs)
    validate_path_conflicts(outputs, protected, label="policy output")
    from sequifier.io.output_paths import validate_training_destinations

    validate_training_destinations(
        config, config.checkpoint_path, protected, phase_offset=config.phase_offset
    )
    require_directory_destinations(archives)
    validate_path_conflicts(archives, protected | set(outputs), label="archive")


def resolve_normalization(policy, declaration):
    real = (
        set(policy.observation_columns + policy.static_columns)
        - policy.categorical_n_classes.keys()
        - set(policy.categorical_columns)
    )
    if declaration is None:
        if real:
            raise ValueError(
                "online-first policy runs require frozen normalization or initialize_from"
            )
        return {}
    if declaration.source == "identity":
        return {column: {"mean": 0.0, "std": 1.0} for column in sorted(real)}
    require_equal(
        sorted(real), sorted(declaration.statistics), "normalization.statistics fields"
    )
    return declaration.statistics


def validate_declared_policy(declared, fitted, field):
    expected, actual = declared.model_dump(mode="json"), fitted.model_dump(mode="json")
    # Unspecified categorical codecs may be fitted once, or supplied by the incoming artifact.
    for key in ("id_maps", "observation_n_classes", "static_n_classes"):
        for column, value in expected.pop(key).items():
            require_equal(value, actual[key].get(column), f"{field}.{key}.{column}")
        actual.pop(key)
    require_equal(expected, actual, field)


def build_actor(config, binding=None, *, artifact=None):
    execution = config.execution
    name = config.policy_interface
    # A runner owns its mutable policy view. Recovery supplies its saved binding.
    interface = execution.interface(name).model_copy(
        update={"policy": binding or config.policy}
    )
    # Exact recovery depends on the selected route being built first.
    interfaces = {name: interface}
    interfaces.update(
        (other, value) for other, value in execution.interfaces.items() if other != name
    )
    execution = replace(execution, interfaces=interfaces)
    network = build_transformer_network(
        execution,
        device=torch.device(config.device),
        initialize=artifact is None,
        restored_history_interface=name
        if config.actor_initialization == "history_only"
        else None,
    ).network
    if artifact is not None:
        from sequifier.artifacts.model_config import load_attached_model_weights

        load_attached_model_weights(network, artifact)
        validate_hash_contracts(network, execution)
    actor = Policy.from_network(network, execution, name).eval()
    actor.units = dict(config.units)
    validate_policy_weights(actor)
    return actor


def validate_data(data, actor):
    validated = set()
    for item in data:
        if item is None or id(item) in validated:
            continue
        validated.add(id(item))
        item.validate(actor)


def preflight_environments(config):
    # Adapter probes must not change the seeded initialization or phase RNG.
    device = torch.device(config.device)
    saved_random = capture_rng_state(device)
    try:
        identities = {}
        for index, phase in enumerate(config.phases, config.phase_offset):
            training_environment = phase.training_environment
            if training_environment is None:
                continue
            require_evaluation_timer()
            for role, spec in (
                ("training", training_environment),
                ("evaluation", phase.evaluation_environment),
            ):
                adapter = load_online_adapter(spec)
                try:
                    if role == "training":
                        if phase.algorithm == "grpo":
                            require_group_start(
                                adapter, exact=config.resume.mode == "exact"
                            )
                            if phase.simulator_workers > 1:
                                from sequifier.rl.simulator_workers import (
                                    require_parallel_group_start,
                                )

                                require_parallel_group_start(adapter)
                        elif config.resume.mode == "exact":
                            require_exact_snapshot(adapter)
                    identities[f"{index}.{role}"] = adapter_identity(adapter)
                finally:
                    adapter.close()
        return identities
    finally:
        restore_rng_state(device, saved_random)


def preflight_optimization(config, actor, data):
    """Resolve constructors and authored scheduler budgets before creating outputs."""
    from sequifier.rl.phase_optimization import PhaseOptimization, PhaseRoles

    baseline = {
        name: parameter.requires_grad
        for name, parameter in actor.network.named_parameters()
    }

    def restore_trainability():
        for name, parameter in actor.network.named_parameters():
            parameter.requires_grad_(baseline[name])

    plans = []
    try:
        for phase, part in zip(config.phases, data):
            restore_trainability()
            roles = PhaseRoles.build(actor, phase, metadata_only=True)
            plan = PhaseOptimization.resolve(
                phase, roles, len(part.training) if part is not None else None
            )
            if phase.optimizer_state == "continue" and plan.masks != plans[-1].masks:
                raise ValueError(
                    "optimizer continuation requires identical resolved parameter masks"
                )
            plan.build(roles)
            plans.append(plan)
    finally:
        restore_trainability()
    return plans
