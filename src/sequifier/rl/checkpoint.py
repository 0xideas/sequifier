"""Versioned local actor weights, input transforms, and policy/return semantics."""

import json
import os
import tempfile
from pathlib import Path

import torch

from sequifier.config.policy_config import CategoricalPolicyConfig
from sequifier.rl.policy import CategoricalPolicy


def validate_policy_weights(policy: CategoricalPolicy) -> None:
    for name, value in policy.state_dict().items():
        if value.device.type != "cpu" or not torch.isfinite(value).all():
            raise ValueError(f"policy tensor {name!r} must be finite and on CPU")
    if (policy.observation_scale <= 0).any() or (policy.static_scale <= 0).any():
        raise ValueError("policy normalization scales must be positive")


def save_policy(
    policy: CategoricalPolicy, path: str | Path, *, metadata: dict | None = None
) -> None:
    validate_policy_weights(policy)
    json.dumps(metadata or {}, allow_nan=False)
    payload = {
        "format": "sequifier.numeric-categorical-policy",
        "version": 1,
        "config": policy.config.model_dump(mode="json"),
        "state_dict": policy.state_dict(),
        "metadata": metadata or {},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, delete=False
        ) as stream:
            temporary = stream.name
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def load_policy(path: str | Path) -> CategoricalPolicy:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "sequifier.numeric-categorical-policy"
        or payload.get("version") != 1
    ):
        raise ValueError("unsupported numeric categorical policy artifact")
    config = CategoricalPolicyConfig.model_validate(payload["config"])
    json.dumps(payload.get("metadata", {}), allow_nan=False)
    with torch.random.fork_rng(devices=[]):
        policy = CategoricalPolicy(config)
    policy.load_state_dict(payload["state_dict"], strict=True)
    validate_policy_weights(policy)
    policy.eval()
    return policy
