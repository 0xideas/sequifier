"""Read-only policy run preparation and whole-plan preflight validation."""

from dataclasses import dataclass, replace

import torch

from sequifier.artifacts.dependencies import (
    DependencyManifest,
    add_backbone_revision,
    policy_dependencies,
)
from sequifier.io.output_paths import validate_training_destinations
from sequifier.io.prepared_policy_data import PreparedPolicyData
from sequifier.model.factory import build_transformer_network, validate_hash_contracts
from sequifier.rl.checkpoint import validate_policy_weights
from sequifier.rl.environment import adapter_identity, load_online_adapter
from sequifier.rl.policy import Policy
from sequifier.rl.ppo_evaluation import require_evaluation_timer
from sequifier.runtime.random_state import capture_rng_state, restore_rng_state


def build_actor(config, *, artifact=None):
    execution = config.parent.execution
    name = config.policy.interface_name
    interface = execution.interface(name)
    # Exact recovery depends on the selected route being built first.
    interfaces = {name: interface}
    interfaces.update(
        (other, value) for other, value in execution.interfaces.items() if other != name
    )
    execution = replace(execution, interfaces=interfaces)
    network = build_transformer_network(
        execution,
        device=torch.device(config.parent.device),
        initialize=artifact is None,
        restored_history_interface=name
        if config.actor_initialization == "history_only"
        else None,
    )
    if artifact is not None:
        from sequifier.artifacts.model_config import load_attached_model_weights

        load_attached_model_weights(network, artifact)
        validate_hash_contracts(network, execution)
    actor = Policy.from_network(network, execution, config.policy).eval()
    validate_policy_weights(actor)
    return actor


def validate_data(phases, data, interface):
    validated = set()
    for phase, item in zip(phases, data, strict=True):
        if item is None or id(item) in validated:
            continue
        try:
            item.validate(interface)
        except (ValueError, KeyError) as error:
            raise ValueError(f"phases.{phase.name}: {error}") from error
        validated.add(id(item))


@dataclass
class PolicyGroupPreparation:
    dependencies: DependencyManifest
    adapters: dict
    data: list[PreparedPolicyData | None]

    @classmethod
    def build(
        cls,
        config,
        prepared,
        *,
        saved,
        committed_reference_count,
        group=False,
        backbone_revision_id=None,
        known_adapters=None,
    ):
        inputs = prepared.resolve(config.parent.policy_data)
        inputs.dependencies.revalidate()
        dependencies = policy_dependencies(
            config,
            prepared,
            inputs,
            saved=saved,
            committed_reference_count=committed_reference_count,
        )
        add_backbone_revision(dependencies, config.parent, backbone_revision_id)
        validate_training_destinations(
            config,
            config.checkpoint_path,
            dependencies.protected_paths,
            phase_offset=config.phase_offset,
            group=group,
        )
        adapters = (
            preflight_environments(config) if known_adapters is None else known_adapters
        )
        data = prepared.prepare(
            config.parent.policy_data, config.phases, config.policy, inputs
        )
        interface = config.parent.execution.interface(config.policy.interface_name)
        validate_data(config.phases, data, interface)
        return cls(dependencies, adapters, data)


def preflight_environments(config):
    from sequifier.rl.registry import BUILTIN_ALGORITHMS

    # Adapter probes must not change the seeded initialization or phase RNG.
    device = torch.device(config.parent.device)
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
                        validate = BUILTIN_ALGORITHMS[phase.algorithm].validate_adapter
                        if validate is not None:
                            validate(adapter, phase, config.resume.mode == "exact")
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
