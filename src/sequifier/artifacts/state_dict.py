"""Canonical model-state naming shared by every artifact boundary."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import TypeVar

import torch
from torch import Tensor

T = TypeVar("T")


def canonical_parameter_name(name: str) -> str:
    """Remove compiler/distributed wrapper segments from one parameter name."""

    parts = [part for part in name.split(".") if part not in {"_orig_mod", "module"}]
    return ".".join(parts)


def canonicalize_state_dict(state_dict: Mapping[str, T]) -> dict[str, T]:
    """Return a canonical state dict and reject ambiguous normalizations."""

    canonical: dict[str, T] = {}
    sources: dict[str, str] = {}
    for name, value in state_dict.items():
        normalized = canonical_parameter_name(name)
        if normalized in canonical:
            raise ValueError(
                "State-dict normalization collision: "
                f"{sources[normalized]!r} and {name!r} both map to {normalized!r}."
            )
        canonical[normalized] = value
        sources[normalized] = name
    return canonical


def validate_model_state_contract(state_dict: Mapping[str, Tensor]) -> None:
    """Validate the portable ``backbone.*``/``interfaces.*`` key contract."""

    invalid = [
        name
        for name in state_dict
        if not (name.startswith("backbone.") or name.startswith("interfaces."))
    ]
    if invalid:
        raise ValueError(
            "Portable model state contains keys outside the stable contract: "
            f"{invalid[:5]!r}."
        )


def weight_fingerprint(weights: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(weights.items()):
        value = tensor.detach().cpu().contiguous()
        if ".prediction_scales.log_scales." in name and value.numel() == 1:
            value = value.reshape(())
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
