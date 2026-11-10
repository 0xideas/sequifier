"""Read-only policy run preparation and whole-plan preflight validation."""

from dataclasses import dataclass
from pathlib import Path

import torch

from sequifier.io.policy_dataset import PolicyDataCatalog
from sequifier.io.prepared_policy_data import PreparedPolicyData
from sequifier.model.factory import build_transformer_network
from sequifier.rl.checkpoint import phase_actor_path, validate_policy_weights
from sequifier.rl.environment import (
    adapter_identity,
    load_online_adapter,
    require_exact_snapshot,
    require_group_start,
)
from sequifier.rl.policy import CategoricalPolicy, policy_execution_config
from sequifier.rl.ppo_evaluation import require_evaluation_timer
from sequifier.rl.registry import BUILTIN_ALGORITHMS
from sequifier.rl.run_state import require_equal
from sequifier.runtime.random_state import capture_rng_state, restore_rng_state
from sequifier.trajectory.sources import (
    ResolvedSequenceSource,
    file_checksum,
    resolve_source,
)


@dataclass(frozen=True)
class RunInputs:
    """Resolved sources and verified file identities owned by one run."""

    sources: dict[str, ResolvedSequenceSource]
    checksums: dict[Path, str]

    @classmethod
    def resolve(cls, config):
        checksums = {}
        sources = {
            name: resolve_source(source, checksums=checksums)
            for name, source in config.sources.items()
        }
        for path in config.preferences.values():
            path = Path(path)
            if path not in checksums:
                checksums[path] = file_checksum(path)
        return cls(sources, checksums)

    @property
    def identity(self):
        return {str(path.resolve()): value for path, value in self.checksums.items()}

    def revalidate(self, expected, field):
        """Rehash every dependency, including manifests, at each phase boundary."""
        require_equal(
            expected,
            {str(path.resolve()): file_checksum(path) for path in self.checksums},
            field,
        )


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
    outputs = [path.resolve() for path in outputs]
    archives = [
        Path(phase.archive_directory).resolve()
        for phase in config.phases
        if getattr(phase, "archive_directory", None)
    ]
    if len(set(outputs)) != len(outputs) or len(set(archives)) != len(archives):
        raise ValueError("policy outputs and phase archives must have distinct paths")
    for path in outputs:
        if path.is_dir() or any(p.exists() and not p.is_dir() for p in path.parents):
            raise ValueError(f"output requires a file destination: {path}")
        for other in protected | set(outputs):
            if (
                path == other
                and other in protected
                or path != other
                and (path in other.parents or other in path.parents)
            ):
                raise ValueError(
                    f"policy output {path} conflicts with input/output {other}"
                )
    for directory in archives:
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"archive requires a directory: {directory}")
        for other in protected | set(outputs) | (set(archives) - {directory}):
            if (
                directory == other
                or directory in other.parents
                or other in directory.parents
            ):
                raise ValueError(
                    f"archive {directory} conflicts with input/output {other}"
                )


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
    actor = CategoricalPolicy.from_network(
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
    catalog = PolicyDataCatalog(inputs.sources, policy, config.episode_cache_size)
    prepared = {}
    result = []
    for phase in config.phases:
        entry = BUILTIN_ALGORITHMS[phase.algorithm]
        if entry.prepare_data is None:
            result.append(None)
            continue
        result.append(entry.prepare_data(config, phase, policy, catalog, prepared))
    return result


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
    budgets = []
    for phase, part in zip(config.phases, data):
        for path in phase.freeze:
            actor.network.get_submodule(path)
        size = len(part.training) if part is not None else None
        budget = phase.optimizer_step_budget(size)
        for optimization in phase.optimization_configs().values():
            optimization.validate_budget(budget)
            optimization.build(actor.network)
        budgets.append(budget)
    return budgets
