"""Canonical hyperparameter-search configuration.

Hyperparameter search always starts from a canonical authored training config
and applies recursive overrides. Historical self-contained search configs and
flat-schema base configs are intentionally unsupported.
"""

from __future__ import annotations

from typing import Any, TypeAlias

from sequifier.config.canonical_hyperparameter_search_config import (
    CanonicalHyperparameterSearchConfig,
    compile_canonical_hyperparameter_search_config,
)
from sequifier.config.composition import load_composed_yaml_config

HyperparameterSearchConfig: TypeAlias = CanonicalHyperparameterSearchConfig
PartialHyperparameterSearchConfig: TypeAlias = CanonicalHyperparameterSearchConfig


def compile_hyperparameter_search_override_config(
    config_path: str,
    config_values: dict[str, Any],
    skip_metadata: bool,
) -> CanonicalHyperparameterSearchConfig:
    """Compatibility name for canonical partial-override compilation."""

    return compile_canonical_hyperparameter_search_config(
        config_path,
        config_values,
        skip_metadata,
    )


def load_hyperparameter_search_config(
    config_path: str,
    skip_metadata: bool,
) -> CanonicalHyperparameterSearchConfig:
    """Load a canonical base-config hyperparameter search."""

    config_values = load_composed_yaml_config(config_path)
    if "overrides" not in config_values:
        raise ValueError(
            f"Hyperparameter search config {config_path!r} must define "
            "'overrides' and reference a canonical training config."
        )
    if not config_values.get("base_config_path"):
        raise ValueError(
            f"Hyperparameter search config {config_path!r} must define a "
            "non-empty 'base_config_path'."
        )
    return compile_canonical_hyperparameter_search_config(
        config_path,
        config_values,
        skip_metadata,
    )


__all__ = [
    "CanonicalHyperparameterSearchConfig",
    "HyperparameterSearchConfig",
    "PartialHyperparameterSearchConfig",
    "compile_canonical_hyperparameter_search_config",
    "compile_hyperparameter_search_override_config",
    "load_hyperparameter_search_config",
]
