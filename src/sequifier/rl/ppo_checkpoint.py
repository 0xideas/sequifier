"""Atomic, algorithm-owned PPO iteration checkpoints."""

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from sequifier.runtime.random_state import RandomState, RandomStateManager


def resume_contract(config):
    values = config.model_dump(mode="json")
    values["training_plan"]["phases"][0].pop("resume")
    return values


def contract_fingerprint(config):
    return hashlib.sha256(
        json.dumps(resume_contract(config), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def clone_weights(module):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def save_ppo_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_ppo_checkpoint(path, config):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("artifact_type") != "sequifier_ppo_run"
        or payload.get("format_version") != 1
    ):
        raise ValueError("PPO resume requires a version-1 PPO learner checkpoint")
    if payload.get("config_fingerprint") != contract_fingerprint(config) or payload.get(
        "config"
    ) != resume_contract(config):
        raise ValueError(
            "PPO resume config, adapter identity, codec, model, or reward scale differs"
        )
    if (
        payload.get("policy") != config.policy.model_dump(mode="json")
        or payload.get("statistics") != config.normalization_statistics
    ):
        raise ValueError("PPO resume frozen representation differs")
    phase = config.training_plan.phases[0]
    steps = payload["env_steps"]
    iteration = payload["iteration"]
    expected_steps = min(iteration * phase.rollout_length, phase.total_env_steps)
    if (
        not isinstance(steps, int)
        or isinstance(steps, bool)
        or not isinstance(iteration, int)
        or isinstance(iteration, bool)
        or iteration < 1
        or not 0 < steps <= phase.total_env_steps
        or steps != expected_steps
        or iteration != (steps + phase.rollout_length - 1) // phase.rollout_length
        or payload["policy_version"] != iteration
        or len(payload["metrics"]) != iteration
    ):
        raise ValueError("PPO checkpoint is not a complete iteration boundary")
    if payload["best_metric"] is not None and not math.isfinite(payload["best_metric"]):
        raise ValueError("PPO checkpoint best metric must be finite or unavailable")
    if (payload["best_actor"] is None) != (payload["best_metric"] is None):
        raise ValueError(
            "PPO checkpoint best actor and metric must be recorded together"
        )
    if set(payload["optimizer_steps"]) != {"actor", "value"} or any(
        not isinstance(clock, int) or isinstance(clock, bool) or clock < iteration
        for clock in payload["optimizer_steps"].values()
    ):
        raise ValueError("PPO checkpoint optimizer clocks are invalid")
    for state in (payload["actor"], payload["value"], payload["best_actor"] or {}):
        if any(not torch.isfinite(tensor).all() for tensor in state.values()):
            raise ValueError("PPO checkpoint contains non-finite weights")
    return payload


def snapshots_equal(left, right):
    """Exact adapters use torch-serializable scalar, tensor and container state."""
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return (
            left.dtype == right.dtype
            and bool(torch.isfinite(left).all())
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, np.ndarray):
        finite = (
            np.isfinite(left).all()
            if left.dtype.kind in "biufc"
            else left.dtype.kind in "US"
        )
        return (
            left.dtype == right.dtype and bool(finite) and np.array_equal(left, right)
        )
    if isinstance(left, np.generic):
        return snapshots_equal(left.item(), right.item())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            snapshots_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            snapshots_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, float):
        return math.isfinite(left) and left == right
    if isinstance(left, (str, int, bool, type(None))):
        return left == right
    raise ValueError(
        "exact adapter snapshots must use scalar, tensor, NumPy array, dict, list, or tuple values"
    )


def capture_rng_state(device):
    return {
        "local": RandomStateManager(device).capture_local().state_dict(),
        "cuda_all": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def restore_rng_state(device, state):
    RandomStateManager(device).restore(RandomState.from_state_dict(state["local"]))
    if state["cuda_all"] is not None:
        if (
            not torch.cuda.is_available()
            or len(state["cuda_all"]) != torch.cuda.device_count()
        ):
            raise ValueError("PPO checkpoint CUDA RNG topology differs")
        torch.cuda.set_rng_state_all(state["cuda_all"])
    if state["mps"] is not None:
        if not torch.backends.mps.is_available():
            raise ValueError("PPO checkpoint requires the saved MPS RNG")
        torch.mps.set_rng_state(state["mps"])
