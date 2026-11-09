"""Sequential policy coordination over prepared data and validated recovery state."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from sequifier.config.train_config import PolicyRunConfig
from sequifier.fingerprints import fingerprint
from sequifier.helpers import configure_determinism
from sequifier.io.preference_dataset import PreferenceData
from sequifier.rl.checkpoint import load_policy, save_policy
from sequifier.rl.offline_data import OfflineData
from sequifier.rl.preparation import (
    build_actor,
    input_identity,
    preflight_environments,
    preflight_optimization,
    prepare_data,
    resolve_normalization,
    validate_data,
    validate_declared_policy,
)
from sequifier.rl.registry import BUILTIN_ALGORITHMS, validate_registry
from sequifier.rl.run_checkpoint import RunCheckpoint
from sequifier.rl.run_state import (
    ActorRepresentation,
    PhaseContext,
    PolicyRunResult,
    capture_rng_state,
    clone_weights,
    require_equal,
    restore_rng_state,
    weight_fingerprint,
    write_json_atomic,
)


def _plan_contract(config):
    values = config.model_dump(mode="json")
    values.pop("resume")
    return values


def run_policy_training(config: PolicyRunConfig) -> PolicyRunResult:
    """Validate the whole run, execute phase-local lifecycles, and export its final actor."""
    if not isinstance(config, PolicyRunConfig):
        raise TypeError("policy training requires PolicyRunConfig")
    validate_registry()
    configure_determinism(config.seed, config.enforce_determinism)
    plan = _plan_contract(config)
    data_identity = input_identity(config)
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
        data = prepare_data(config, representation.policy)
        actor = build_actor(
            config, representation.policy, representation.values["statistics"]
        )
        recovery.validate_actor(actor, config.units)
    else:
        data = prepare_data(config, config.policy)
        if config.initialize_from:
            source = load_policy(
                config.initialize_from,
                device=config.device,
                interface_name=config.policy_interface,
            )
            validate_declared_policy(
                config.policy, source.config, "initialize_from.policy"
            )
            actor = build_actor(
                config, source.config, source.interface.selected_columns_statistics
            )
            ActorRepresentation.from_actor(actor, config.units).validate_actor(
                source, config.units, "initialize_from"
            )
            actor.network.load_state_dict(source.network.state_dict(), strict=True)
        else:
            first = data[0]
            fitting = (
                first.training.fitting_dataset()
                if isinstance(first, PreferenceData)
                else first.training
                if isinstance(first, OfflineData)
                else None
            )
            fitted = (
                fitting.fit_categorical_config()
                if fitting is not None
                else config.policy
            )
            if set(fitted.categorical_columns) - fitted.categorical_n_classes.keys():
                raise ValueError(
                    "online-first categorical fields require frozen class counts and codecs"
                )
            if config.normalization is not None or fitting is None:
                statistics = resolve_normalization(fitted, config.normalization)
            else:
                fitting.use_config(fitted)
                statistics = fitting.normalization_statistics()
            actor = build_actor(config, fitted, statistics)
        representation = ActorRepresentation.from_actor(actor, config.units)
    if config.normalization is not None:
        require_equal(
            resolve_normalization(actor.config, config.normalization),
            representation.values["statistics"],
            "run.normalization",
        )
    validate_data(config, data, actor)
    data_contract = [item.contract if item is not None else None for item in data]
    if recovery is not None:
        require_equal(data_contract, recovery.data_contract, "resume.partitions")
    for phase in config.phases:
        if phase.representation is not None:
            require_equal(
                representation.values["policy"],
                phase.representation.model_dump(mode="json"),
                f"phases.{phase.name}.representation",
            )
        normalization = getattr(phase, "normalization", None)
        if normalization is not None:
            require_equal(
                representation.values["statistics"],
                resolve_normalization(actor.config, normalization),
                f"phases.{phase.name}.normalization",
            )
    if recovery is None:
        recovery = RunCheckpoint(
            plan,
            representation,
            data_identity,
            data_contract,
            adapter_contract,
            {
                "path": config.initialize_from,
                "weight_hash": weight_fingerprint(actor.network.state_dict()),
            },
            clone_weights(actor.network),
            capture_rng_state(actor.device),
        )
    initialization = recovery.initialization
    if resuming and config.initialize_from and Path(config.initialize_from).is_file():
        initial = load_policy(
            config.initialize_from,
            device=config.device,
            interface_name=config.policy_interface,
        )
        representation.validate_actor(initial, config.units, "resume.initialize_from")
        require_equal(
            initialization["weight_hash"],
            weight_fingerprint(initial.network.state_dict()),
            "resume.initialize_from.weights",
        )
    reference_contract = recovery.reference_contract
    for index, phase in enumerate(config.phases):
        reference = getattr(phase, "reference", None)
        if reference is None or reference.source != "artifact":
            continue
        path = Path(reference.path)
        if path.is_file():
            ref = load_policy(
                path, device=config.device, interface_name=config.policy_interface
            )
            representation.validate_actor(
                ref, config.units, f"phases.{phase.name}.reference"
            )
            identity = weight_fingerprint(ref.network.state_dict())
            if str(index) in reference_contract:
                require_equal(
                    reference_contract[str(index)],
                    identity,
                    f"phases.{phase.name}.reference.weights",
                )
            reference_contract[str(index)] = identity
        elif (
            index > recovery.phase_index
            or index == recovery.phase_index
            and recovery.phase_state is None
        ):
            raise FileNotFoundError(
                f"reference artifact required before phase start: {path}"
            )
    saved_random = capture_rng_state(actor.device)
    optimization = preflight_optimization(config, actor, data)
    restore_rng_state(actor.device, recovery.random if resuming else saved_random)

    published = set()

    def publish_completed():
        for result, weights in zip(recovery.results, recovery.selected_actors):
            if result.index in published:
                continue
            portable = deepcopy(actor).eval()
            portable.network.load_state_dict(weights, strict=True)
            directory = (
                Path(config.phase_directory) / f"{result.index:03d}-{result.name}"
            )
            save_policy(portable, directory / "actor.pt", metadata=result.as_dict())
            write_json_atomic(directory / "report.json", result.as_dict())
            published.add(result.index)

    if not resuming:
        # Persist the complete fitted representation before the first update.
        recovery.save(checkpoint, actor)
    while recovery.phase_index < len(config.phases):
        phase_index = recovery.phase_index
        current_state = recovery.phase_state
        phase = config.phases[phase_index]
        require_equal(
            data_identity,
            input_identity(config),
            f"phases.{phase.name}.source_identity",
        )
        incoming_weight_hash = weight_fingerprint(actor.network.state_dict())
        phase_actor = deepcopy(actor).eval()
        reference, reference_identity = None, None
        declaration = getattr(phase, "reference", None)
        if declaration is not None:
            if current_state is not None:
                reference = deepcopy(phase_actor)
                reference.network.load_state_dict(
                    current_state["reference"], strict=True
                )
            elif declaration.source == "artifact":
                reference = load_policy(
                    declaration.path,
                    device=config.device,
                    interface_name=config.policy_interface,
                )
            else:
                reference = deepcopy(phase_actor)
            reference.eval().requires_grad_(False)
            representation.validate_actor(
                reference, config.units, f"phases.{phase.name}.reference"
            )
            reference_identity = weight_fingerprint(reference.network.state_dict())
            if current_state is not None:
                require_equal(
                    reference_identity,
                    current_state["reference_identity"],
                    "resume.reference.weights",
                )
            if declaration.source == "artifact":
                require_equal(
                    reference_contract[str(phase_index)],
                    reference_identity,
                    "reference.artifact.weights",
                )

        def commit(state):
            representation.validate_actor(
                phase_actor, config.units, f"phases.{phase.name}.actor"
            )
            recovery.save(checkpoint, actor, state)

        context = PhaseContext(
            config,
            phase_index,
            phase,
            phase_actor,
            representation,
            data[phase_index],
            reference,
            reference_identity,
            current_state,
            commit,
            optimization[phase_index],
        )
        actor, result = BUILTIN_ALGORITHMS[phase.algorithm].runtime(context)
        representation.validate_actor(
            actor, config.units, f"phases.{phase.name}.selected_actor"
        )
        result = replace(
            result,
            provenance={
                **result.provenance,
                "config_fingerprint": fingerprint(phase.model_dump(mode="json")),
                "representation_fingerprint": representation.identity,
                "incoming_weight_hash": incoming_weight_hash,
                "actor_path": str(
                    Path(config.phase_directory)
                    / f"{phase_index:03d}-{phase.name}"
                    / "actor.pt"
                ),
                "report_path": str(
                    Path(config.phase_directory)
                    / f"{phase_index:03d}-{phase.name}"
                    / "report.json"
                ),
            },
        )
        recovery.finish_phase(actor, result)
        recovery.save(checkpoint, actor)
        publish_completed()
    publish_completed()
    report = {
        "format": "sequifier.policy-run-report",
        "version": 1,
        "plan": plan,
        "plan_fingerprint": fingerprint(plan),
        "representation_fingerprint": representation.identity,
        "source_provenance": data_identity,
        "reference_provenance": reference_contract,
        "initialization": initialization,
        "resume": config.resume.model_dump(mode="json"),
        "phases": [r.as_dict() for r in recovery.results],
        "model_path": config.model_path,
        "checkpoint_path": config.checkpoint_path,
    }
    save_policy(actor.eval(), config.model_path, metadata=report)
    write_json_atomic(config.report_path, report)
    recovery.save(checkpoint, actor)
    return PolicyRunResult(
        actor,
        tuple(recovery.results),
        representation,
        config.model_path,
        config.report_path,
        config.checkpoint_path,
    )
