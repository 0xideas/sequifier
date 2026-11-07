"""Validated seed windows and atomic numeric feature/mask advancement."""

import math
from collections.abc import Mapping
from numbers import Real

from sequifier.trajectory.schema import SeedHistory


def numeric_scalar(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite numeric scalar, got {value!r}")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{label} is outside the supported numeric range") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def numeric_fields(values, expected: set[str], label: str) -> dict[str, float]:
    if not isinstance(values, Mapping) or set(values) != expected:
        raise ValueError(
            f"{label} must supply exactly these fields: {sorted(expected)}"
        )
    return {
        key: numeric_scalar(values[key], f"{label}.{key}") for key in sorted(expected)
    }


def validate_seed(seed: SeedHistory) -> SeedHistory:
    return SeedHistory.model_validate(seed.model_dump())


def advance_history(history: SeedHistory, successor: dict[str, float]) -> SeedHistory:
    return SeedHistory(
        observations={
            key: values[1:] + [successor[key]]
            for key, values in history.observations.items()
        },
        attention_valid_mask=history.attention_valid_mask[1:] + [True],
        static_inputs=dict(history.static_inputs),
    )
