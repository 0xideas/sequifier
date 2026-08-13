"""Validated controller directives for the training engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch.optim import Optimizer

_SUPPORTED_GROUP_FIELDS = frozenset({"lr", "weight_decay", "momentum", "betas"})


@dataclass(frozen=True)
class TrainingDirective:
    parameter_group_updates: dict[str, dict[str, Any]] = field(default_factory=dict)
    gradient_clip_norm: float | None = None
    skip_optimizer_step: bool = False
    stop_after_step: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.gradient_clip_norm is not None and self.gradient_clip_norm < 0:
            raise ValueError("gradient_clip_norm must be non-negative.")
        for group_id, updates in self.parameter_group_updates.items():
            if not group_id:
                raise ValueError("Optimizer group IDs must not be empty.")
            unsupported = set(updates).difference(_SUPPORTED_GROUP_FIELDS)
            if unsupported:
                raise ValueError(
                    f"Unsupported optimizer group fields for {group_id!r}: "
                    f"{sorted(unsupported)!r}."
                )


def apply_training_directive(
    optimizer: Optimizer, directive: TrainingDirective
) -> None:
    """Validate and apply declared optimizer-group changes atomically."""

    groups = {
        str(group.get("group_id", "all")): group for group in optimizer.param_groups
    }
    unknown_groups = set(directive.parameter_group_updates).difference(groups)
    if unknown_groups:
        raise ValueError(
            f"Unknown optimizer group IDs: {sorted(unknown_groups)!r}; "
            f"available groups are {sorted(groups)!r}."
        )

    validated: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for group_id, updates in directive.parameter_group_updates.items():
        group = groups[group_id]
        normalized = dict(updates)
        for field_name in ("lr", "weight_decay", "momentum"):
            if field_name in normalized:
                value = float(normalized[field_name])
                if value < 0:
                    raise ValueError(
                        f"{field_name} for {group_id!r} must be non-negative."
                    )
                if field_name == "momentum" and field_name not in group:
                    raise ValueError(
                        f"Optimizer group {group_id!r} does not support momentum."
                    )
                normalized[field_name] = value
        if "betas" in normalized:
            if "betas" not in group:
                raise ValueError(
                    f"Optimizer group {group_id!r} does not support betas."
                )
            betas = tuple(float(value) for value in normalized["betas"])
            if len(betas) != 2 or any(value < 0 or value >= 1 for value in betas):
                raise ValueError("betas must contain two values in [0, 1).")
            normalized["betas"] = betas
        validated.append((group, normalized))

    for group, updates in validated:
        group.update(updates)
