"""Domain scalars shared by table import, preprocessing and environment adapters."""

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Annotated, Any, TypeAlias

from pydantic import BeforeValidator, FiniteFloat, StrictInt, StrictStr

DomainValue: TypeAlias = Annotated[
    StrictInt | FiniteFloat | StrictStr,
    BeforeValidator(lambda value: domain_scalar(value, "domain value")),
]


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


def portable_value(value):
    """Own JSON-compatible copies of array, tensor, and nested input payloads."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: portable_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("model payloads must contain finite values")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"unsupported model payload type: {type(value).__name__}")


def domain_scalar(value, label: str):
    if isinstance(value, str):
        return value
    number = numeric_scalar(value, label)
    return int(value) if isinstance(value, Integral) else number


def domain_fields(values, expected: set[str], label: str) -> dict:
    if not isinstance(values, Mapping) or set(values) != expected:
        raise ValueError(
            f"{label} must supply exactly these fields: {sorted(expected)}"
        )
    return {
        key: domain_scalar(values[key], f"{label}.{key}") for key in sorted(expected)
    }


def numeric_fields(values, expected: set[str], label: str) -> dict[str, float]:
    return {
        key: numeric_scalar(value, f"{label}.{key}")
        for key, value in domain_fields(values, expected, label).items()
    }


def portable_fields(values: Mapping[str, Any]) -> dict[str, Any]:
    return {name: portable_value(value) for name, value in values.items()}
