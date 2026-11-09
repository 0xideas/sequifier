"""Sequential policy coordinator and the sole versioned policy run checkpoint."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import torch

from sequifier.config.train_config import PolicyRunConfig
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
from sequifier.rl.run_state import (
    ActorRepresentation,
    PhaseContext,
    PolicyPhaseResult,
    PolicyRunResult,
    atomic_save,
    capture_rng_state,
    clone_weights,
    fingerprint,
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
    if config.resume.policy == "required" and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = None
    if config.resume.policy != "never" and checkpoint.is_file():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, dict)
            or payload.get("artifact_type") != "sequifier_policy_run"
            or payload.get("format_version") != 1
        ):
            raise ValueError(
                "policy resume requires a version-1 multi-phase policy run checkpoint"
            )
        require_equal(plan, payload["plan"], "resume.plan")
        require_equal(
            fingerprint(plan), payload["plan_fingerprint"], "resume.plan_fingerprint"
        )
        require_equal(data_identity, payload["data_identity"], "resume.data")
    adapter_contract = preflight_environments(config)
    if payload:
        require_equal(
            adapter_contract, payload["adapter_contract"], "resume.environments"
        )
        representation = ActorRepresentation(payload["representation"])
        require_equal(
            representation.identity,
            payload["representation_fingerprint"],
            "resume.representation_fingerprint",
        )
        validate_declared_policy(config.policy, representation.policy, "resume.policy")
        data = prepare_data(config, representation.policy)
        actor = build_actor(
            config, representation.policy, representation.values["statistics"]
        )
        representation.validate_actor(actor, config.units, "resume.representation")
        require_equal(
            weight_fingerprint(payload["run_actor"]),
            payload["run_actor_identity"],
            "resume.incoming_actor.weights",
        )
        actor.network.load_state_dict(payload["run_actor"], strict=True)
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
                first.training.fitting_dataset(config.episode_cache_size)
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
    if payload:
        require_equal(data_contract, payload["data_contract"], "resume.partitions")
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
    phase_index = payload["phase_index"] if payload else 0
    results = (
        [
            PolicyPhaseResult(**{**result, "metrics": tuple(result["metrics"])})
            for result in payload["completed_results"]
        ]
        if payload
        else []
    )
    selected_actors = payload["selected_actors"] if payload else []
    current_state = payload["phase_state"] if payload else None
    if (
        not isinstance(phase_index, int)
        or not 0 <= phase_index <= len(config.phases)
        or len(results) != phase_index
        or len(selected_actors) != phase_index
    ):
        raise ValueError("checkpoint phase index/completed handoffs disagree")
    if payload and payload["status"] not in {"ready", "running", "handoff", "complete"}:
        raise ValueError("invalid policy checkpoint status")
    if payload:
        status = payload["status"]
        if (
            (status == "running") != (current_state is not None)
            or status == "complete"
            and phase_index != len(config.phases)
            or status == "ready"
            and phase_index != 0
            or status == "handoff"
            and phase_index == 0
        ):
            raise ValueError(
                "checkpoint status disagrees with the committed phase boundary"
            )
        require_equal(
            {
                "phase_index": phase_index,
                "transaction": current_state["transaction"] if current_state else None,
            },
            payload["last_committed_transaction"],
            "resume.last_committed_transaction",
        )
    if current_state is not None:
        if phase_index == len(config.phases):
            raise ValueError("completed run cannot have a live phase state")
        phase = config.phases[phase_index]
        entry = BUILTIN_ALGORITHMS[phase.algorithm]
        require_equal(
            {"algorithm": phase.algorithm, "version": entry.state_version},
            payload["phase_runtime"] if payload else None,
            "resume.phase_runtime",
        )
        if current_state.get("version") != entry.state_version:
            raise ValueError("unsupported phase checkpoint state version")
    if current_state is not None:
        restored_actor = deepcopy(actor)
        restored_actor.network.load_state_dict(current_state["actor"], strict=True)
        if any(
            not torch.isfinite(t).all()
            for t in restored_actor.network.state_dict().values()
        ):
            raise ValueError("non-finite checkpoint actor")
        if getattr(config.phases[phase_index], "reference", None) is not None:
            reference_weights = current_state["reference"]
            restored_actor.network.load_state_dict(reference_weights, strict=True)
            require_equal(
                weight_fingerprint(reference_weights),
                current_state["reference_identity"],
                "resume.embedded_reference",
            )
            if any(not torch.isfinite(t).all() for t in reference_weights.values()):
                raise ValueError("non-finite checkpoint reference")
    initialization = (
        payload["initialization"]
        if payload
        else {
            "path": config.initialize_from,
            "weight_hash": weight_fingerprint(actor.network.state_dict()),
        }
    )
    if payload and config.initialize_from and Path(config.initialize_from).is_file():
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
    if (
        current_state is not None
        and getattr(config.phases[phase_index], "reference", None) is not None
    ):
        declaration = getattr(config.phases[phase_index], "reference")
        if declaration.source == "phase_start":
            require_equal(
                weight_fingerprint(current_state["reference"]),
                weight_fingerprint(actor.network.state_dict()),
                "resume.phase_start_reference.weights",
            )
    reference_contract = dict(payload["reference_contract"]) if payload else {}
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
            not payload
            or index > phase_index
            or index == phase_index
            and current_state is None
        ):
            raise FileNotFoundError(
                f"reference artifact required before phase start: {path}"
            )
    for index, (result, weights) in enumerate(zip(results, selected_actors)):
        require_equal(index, result.index, "resume.completed_phase.index")
        require_equal(
            config.phases[index].name, result.name, "resume.completed_phase.name"
        )
        require_equal(
            config.phases[index].algorithm,
            result.algorithm,
            "resume.completed_phase.algorithm",
        )
        require_equal(
            result.selected_weight_hash,
            weight_fingerprint(weights),
            "resume.selected_actor.weights",
        )
        if any(not torch.isfinite(t).all() for t in weights.values()):
            raise ValueError("non-finite completed phase actor")
    if selected_actors:
        require_equal(
            weight_fingerprint(actor.network.state_dict()),
            weight_fingerprint(selected_actors[-1]),
            "resume.incoming_selected_actor",
        )
    saved_random = capture_rng_state(actor.device)
    preflight_optimization(config, actor, data)
    restore_rng_state(actor.device, payload["random"] if payload else saved_random)

    def envelope(status, state=None):
        runtime = (
            None
            if phase_index == len(config.phases)
            else {
                "algorithm": config.phases[phase_index].algorithm,
                "version": BUILTIN_ALGORITHMS[
                    config.phases[phase_index].algorithm
                ].state_version,
            }
        )
        return {
            "artifact_type": "sequifier_policy_run",
            "format_version": 1,
            "plan": plan,
            "plan_fingerprint": fingerprint(plan),
            "representation": representation.canonical,
            "representation_fingerprint": representation.identity,
            "data_identity": data_identity,
            "data_contract": data_contract,
            "adapter_contract": adapter_contract,
            "reference_contract": reference_contract,
            "random": capture_rng_state(actor.device),
            "phase_index": phase_index,
            "status": status,
            "completed_results": [r.as_dict() for r in results],
            "selected_actors": selected_actors,
            "run_actor": clone_weights(actor.network),
            "run_actor_identity": weight_fingerprint(actor.network.state_dict()),
            "initialization": initialization,
            "resume_request": config.resume.model_dump(mode="json"),
            "phase_runtime": runtime,
            "phase_state": state,
            "last_committed_transaction": {
                "phase_index": phase_index,
                "transaction": state["transaction"] if state else None,
            },
        }

    published = set()

    def publish_completed():
        for result, weights in zip(results, selected_actors):
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

    if payload is None:
        # Persist the complete fitted representation before the first update.
        atomic_save(checkpoint, envelope("ready"))
    while phase_index < len(config.phases):
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
            atomic_save(checkpoint, envelope("running", state))

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
        results.append(result)
        selected_actors.append(clone_weights(actor.network))
        phase_index += 1
        current_state = None
        atomic_save(checkpoint, envelope("handoff"))
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
        "phases": [r.as_dict() for r in results],
        "model_path": config.model_path,
        "checkpoint_path": config.checkpoint_path,
    }
    save_policy(actor.eval(), config.model_path, metadata=report)
    write_json_atomic(config.report_path, report)
    atomic_save(checkpoint, envelope("complete"))
    return PolicyRunResult(
        actor,
        tuple(results),
        representation,
        config.model_path,
        config.report_path,
        config.checkpoint_path,
    )
