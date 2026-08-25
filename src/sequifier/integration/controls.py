from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

from torch.optim import Optimizer

from sequifier.typechecking import beartype

_PROTECTED_GROUP_FIELDS = frozenset({"params", "group_id"})
_NON_NEGATIVE_GROUP_FIELDS = frozenset({"lr", "weight_decay", "momentum", "eps"})


@dataclass(frozen=True)
class TrainingDirective:
    parameter_group_updates: dict[str, dict[str, Any]] = field(default_factory=dict)
    gradient_clip_norm: float | None = None
    skip_optimizer_step: bool = False
    stop_after_step: bool = False
    reason: str | None = None
    scheduler_state_updates: dict[str, Any] = field(default_factory=dict)
    disable_gradient_clipping: bool = False
    skip_scheduler_step: bool = False

    @beartype
    def __post_init__(self) -> None:
        if self.gradient_clip_norm is not None and self.gradient_clip_norm < 0:
            raise ValueError("gradient_clip_norm must be non-negative.")
        if self.gradient_clip_norm is not None and self.disable_gradient_clipping:
            raise ValueError(
                "gradient_clip_norm and disable_gradient_clipping cannot both be set."
            )
        for group_id, updates in self.parameter_group_updates.items():
            if not group_id:
                raise ValueError("Optimizer group IDs must not be empty.")
            protected = set(updates).intersection(_PROTECTED_GROUP_FIELDS)
            if protected:
                raise ValueError(
                    f"Protected optimizer group fields for {group_id!r} cannot be "
                    f"updated: {sorted(protected)!r}."
                )
            if not all(isinstance(name, str) and name for name in updates):
                raise ValueError("Optimizer group field names must not be empty.")
        if not all(
            isinstance(name, str) and name for name in self.scheduler_state_updates
        ):
            raise ValueError("Scheduler state field names must not be empty.")


@beartype
def _normalize_sequence_update(
    *, group_id: str, field_name: str, value: Any, current: tuple[Any, ...]
) -> tuple[Any, ...]:
    try:
        normalized = tuple(value)
    except TypeError as error:
        raise TypeError(
            f"{field_name} for optimizer group {group_id!r} must be a sequence."
        ) from error
    if len(normalized) != len(current):
        raise ValueError(
            f"{field_name} for optimizer group {group_id!r} must contain "
            f"{len(current)} values."
        )
    if field_name == "betas":
        betas = tuple(float(item) for item in normalized)
        if any(not math.isfinite(item) or item < 0 or item >= 1 for item in betas):
            raise ValueError("betas values must be finite and in [0, 1).")
        return betas
    return normalized


@beartype
def _normalize_group_update(
    *, group_id: str, field_name: str, value: Any, current: Any
) -> Any:
    if field_name in _NON_NEGATIVE_GROUP_FIELDS:
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0:
            raise ValueError(
                f"{field_name} for optimizer group {group_id!r} must be a "
                "finite, non-negative value."
            )
        return normalized
    if isinstance(current, tuple):
        return _normalize_sequence_update(
            group_id=group_id,
            field_name=field_name,
            value=value,
            current=current,
        )
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise TypeError(
                f"{field_name} for optimizer group {group_id!r} must be a bool."
            )
        return value
    if isinstance(current, Integral):
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise TypeError(
                f"{field_name} for optimizer group {group_id!r} must be an integer."
            )
        return int(value)
    if isinstance(current, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(
                f"{field_name} for optimizer group {group_id!r} must be finite."
            )
        return normalized
    if current is not None and not isinstance(value, type(current)):
        raise TypeError(
            f"{field_name} for optimizer group {group_id!r} must have type "
            f"{type(current).__name__}."
        )
    return value


@beartype
def apply_training_directive(
    optimizer: Optimizer,
    directive: TrainingDirective,
    scheduler: Any | None = None,
) -> None:
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
        unknown_fields = set(updates).difference(group)
        if unknown_fields:
            raise ValueError(
                f"Optimizer group {group_id!r} does not expose fields "
                f"{sorted(unknown_fields)!r}."
            )
        normalized = {
            field_name: _normalize_group_update(
                group_id=group_id,
                field_name=field_name,
                value=value,
                current=group[field_name],
            )
            for field_name, value in updates.items()
        }
        validated.append((group, normalized))

    scheduler_state = None
    scheduler_state_before = None
    scheduler_to_update = None
    if directive.scheduler_state_updates:
        if scheduler is None:
            raise ValueError(
                "scheduler_state_updates require an initialized scheduler."
            )
        scheduler_to_update = scheduler
        scheduler_state_before = copy.deepcopy(scheduler.state_dict())
        unknown_scheduler_fields = set(directive.scheduler_state_updates).difference(
            scheduler_state_before
        )
        if unknown_scheduler_fields:
            raise ValueError(
                "Scheduler does not expose state fields "
                f"{sorted(unknown_scheduler_fields)!r}."
            )
        scheduler_state = copy.deepcopy(scheduler_state_before)
        scheduler_state.update(copy.deepcopy(directive.scheduler_state_updates))

    group_state_before = [
        (group, {field_name: group[field_name] for field_name in updates})
        for group, updates in validated
    ]
    try:
        for group, updates in validated:
            group.update(updates)
        if scheduler_state is not None:
            if scheduler_to_update is None:
                raise RuntimeError("Validated scheduler reference is unavailable.")
            scheduler_to_update.load_state_dict(scheduler_state)
    except Exception:
        for group, previous in group_state_before:
            group.update(previous)
        if scheduler_state_before is not None:
            if scheduler_to_update is None:
                raise RuntimeError("Validated scheduler reference is unavailable.")
            scheduler_to_update.load_state_dict(scheduler_state_before)
        raise
