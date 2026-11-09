"""Read-only policy run preparation and whole-plan preflight validation."""

import hashlib
from pathlib import Path

import torch

from sequifier.io.policy_dataset import PolicyDataCatalog
from sequifier.io.policy_window import PolicyWindowView
from sequifier.io.preference_dataset import PreferenceData
from sequifier.model.factory import build_transformer_network
from sequifier.rl.environment import (
    adapter_identity,
    load_online_adapter,
    require_exact_snapshot,
    require_group_start,
)
from sequifier.rl.offline_data import OfflineData
from sequifier.rl.policy import CategoricalPolicy, policy_execution_config
from sequifier.rl.ppo_evaluation import require_evaluation_timer
from sequifier.rl.registry import BUILTIN_ALGORITHMS
from sequifier.rl.run_state import capture_rng_state, require_equal, restore_rng_state
from sequifier.trajectory.sources import source_paths


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_identity(config):
    protected = set()
    for source in config.sources.values():
        protected.update(source_paths(source))
        protected.update(
            Path(path).resolve()
            for path in (
                source.manifest,
                source.metadata_config_path,
                source.static_data_path,
            )
            if path
        )
    protected.update(Path(path) for path in config.preferences.values())
    data_identity = {str(path): file_hash(path) for path in sorted(protected)}
    if config.initialize_from:
        protected.add(Path(config.initialize_from))
    protected.update(
        Path(phase.reference.path)
        for phase in config.phases
        if getattr(phase, "reference", None) is not None and phase.reference.path
    )
    outputs = [
        Path(config.model_path),
        Path(config.report_path),
        Path(config.checkpoint_path),
    ]
    for i, phase in enumerate(config.phases):
        directory = Path(config.phase_directory) / f"{i:03d}-{phase.name}"
        outputs += [directory / "actor.pt", directory / "report.json"]
    archives = [
        Path(phase.archive_directory)
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
    return data_identity


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
    actor.set_normalization_statistics(statistics)
    if any(
        (tensor.is_floating_point() and tensor.dtype != torch.float32)
        or not torch.isfinite(tensor).all()
        for tensor in actor.network.state_dict().values()
    ):
        raise ValueError("policy initialization requires finite FP32 network weights")
    return actor


def prepare_data(config, policy):
    catalog = PolicyDataCatalog(config.sources, policy, config.episode_cache_size)
    prepared = {}
    result = []
    for phase in config.phases:
        entry = BUILTIN_ALGORITHMS[phase.algorithm]
        if entry.prepare_data is None:
            result.append(None)
            continue
        result.append(entry.prepare_data(config, phase, policy, catalog, prepared))
    return result


def validate_data(config, data, actor):
    policy = actor.config
    validated = set()
    for item in data:
        if id(item) in validated:
            continue
        validated.add(id(item))
        if isinstance(item, OfflineData):
            item.use_config(policy)
            for index, part in enumerate((item.training, item.validation)):
                part.validate_normalization(actor.interface.selected_columns_statistics)
                if item.transitions is not None:
                    transitions = item.transitions[index]
                    for episode in part.episodes:
                        step = len(episode.actions) - 1
                        successor = transitions._successor(episode, step)
                        if successor is not None:
                            inputs = PolicyWindowView(policy).successor_inputs(
                                episode, step, successor
                            )
                            features, _ = actor.model_inputs(inputs.to(actor.device))
                            if any(
                                t.is_floating_point() and not torch.isfinite(t).all()
                                for t in features.values()
                            ):
                                raise ValueError(
                                    f"episode {episode.episode_id}: non-finite normalized IQL successor"
                                )
        elif isinstance(item, PreferenceData):
            item.training.use_config(policy)
            item.validation.use_config(policy)
            for partition in (item.training, item.validation):
                fitting = partition.fitting_dataset()
                fitting.use_config(policy)
                fitting.validate_normalization(
                    actor.interface.selected_columns_statistics
                )


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
    plans = []
    for phase, part in zip(config.phases, data):
        size = len(part.training) if part is not None else None
        roles = phase.optimization_plans(size)
        for plan in roles.values():
            plan.build(actor)
        plans.append(roles)
    return plans
