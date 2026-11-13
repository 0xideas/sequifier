"""Read-only policy run preparation and whole-plan preflight validation."""

from pathlib import Path

import torch

from sequifier.io.output_paths import (
    require_directory_destinations,
    require_file_destinations,
    unique_paths,
    validate_path_conflicts,
)
from sequifier.io.prepared_policy_data import PreparedPolicyData
from sequifier.model.factory import build_transformer_network
from sequifier.rl.checkpoint import phase_actor_path, validate_policy_weights
from sequifier.rl.environment import (
    adapter_identity,
    load_online_adapter,
    require_exact_snapshot,
    require_group_start,
)
from sequifier.rl.policy import Policy, policy_execution_config
from sequifier.rl.ppo_evaluation import require_evaluation_timer
from sequifier.rl.prepared_inputs import PreparedTrainingInputs, RunInputs  # noqa: F401
from sequifier.rl.run_state import require_equal
from sequifier.runtime.random_state import capture_rng_state, restore_rng_state


def validate_output_paths(config, inputs):
    """Keep every output and archive separate from inputs and other outputs."""
    protected = {path.resolve() for path in inputs.checksums}
    if config.initialize_from:
        protected.add(Path(config.initialize_from).resolve())
    protected.update(
        Path(phase.reference.path).resolve()
        for phase in config.phases
        if getattr(phase, "reference", None) is not None and phase.reference.path
    )
    outputs = [
        Path(config.model_path),
        Path(config.report_path),
        Path(config.checkpoint_path),
    ]
    for i, phase in enumerate(config.phases):
        actor = phase_actor_path(config.phase_directory, i, phase.name)
        outputs += [actor, actor.with_name("report.json")]
    message = "policy outputs and phase archives must have distinct paths"
    outputs = unique_paths(outputs, message)
    archives = unique_paths(
        [
            Path(phase.archive_directory).resolve()
            for phase in config.phases
            if getattr(phase, "archive_directory", None)
        ],
        message,
    )
    require_file_destinations(outputs)
    validate_path_conflicts(outputs, protected, label="policy output")
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


def build_actor(config, fitted, statistics):
    execution = policy_execution_config(
        fitted,
        config.model,
        config.device,
        statistics=statistics,
        execution_model=config.execution_model,
        project_root=config.project_root,
        model_name=config.model_name,
        seed=config.seed,
    )
    execution.embedding_layer_names = config.embedding_layer_names
    actor = Policy.from_network(
        build_transformer_network(
            execution, device=torch.device(config.device)
        ).network,
        execution,
        config.policy_interface,
    ).eval()
    actor.units = dict(config.units)
    validate_policy_weights(actor)
    return actor


def prepare_data(config, policy, inputs) -> list[PreparedPolicyData | None]:
    return PreparedTrainingInputs.for_run(config).prepare(config, policy, inputs)


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
        for index, phase in enumerate(config.phases):
            if not hasattr(phase, "environment"):
                continue
            require_evaluation_timer()
            for role, spec in (
                ("training", phase.environment),
                ("evaluation", phase.evaluation.environment),
            ):
                adapter = load_online_adapter(spec)
                try:
                    if role == "training":
                        if phase.algorithm == "grpo":
                            require_group_start(
                                adapter, exact=config.resume.mode == "exact"
                            )
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
    from copy import deepcopy

    from sequifier.rl.models import HistoryCritic, make_iql_models

    baseline = {
        name: parameter.requires_grad
        for name, parameter in actor.network.named_parameters()
    }
    budgets, previous_masks = [], None
    try:
        for phase, part in zip(config.phases, data):
            for name, parameter in actor.network.named_parameters():
                parameter.requires_grad_(baseline[name])
            modules = {"actor": actor.network}
            if phase.algorithm == "iql":
                value, q1, q2, _ = make_iql_models(
                    actor.execution_config, actor.interface_name, actor.config.actions
                )
                modules.update(value=value, critic=torch.nn.ModuleList([q1, q2]))
            elif phase.algorithm == "ppo":
                modules["value"] = HistoryCritic(
                    deepcopy(actor.execution_config), actor.interface_name, None
                )
            masks = {}
            for role, module in modules.items():
                frozen = phase.trainability.policy(role).resolve(module)
                for parameter in module.parameters():
                    if id(parameter) in frozen:
                        parameter.requires_grad_(False)
                masks[role] = sorted(
                    name
                    for name, parameter in module.named_parameters()
                    if parameter.requires_grad
                )
                if not masks[role]:
                    raise ValueError(
                        f"{phase.name}: {role} leaves no trainable parameters"
                    )
            if phase.optimizer_state == "continue" and masks != previous_masks:
                raise ValueError(
                    "optimizer continuation requires identical resolved parameter masks"
                )
            previous_masks = masks
            size = len(part.training) if part is not None else None
            budget = phase.optimizer_step_budget(size)
            for role, optimization in phase.optimization_configs().items():
                optimization.validate_budget(budget)
                optimization.build(modules[role], step_budget=budget)
            budgets.append(budget)
    finally:
        for name, parameter in actor.network.named_parameters():
            parameter.requires_grad_(baseline[name])
    return budgets
