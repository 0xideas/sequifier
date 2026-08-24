"""Compatibility facade for canonical partial hyperparameter searches.

Canonical partial searches reference a canonical training config through
``base_config_path`` and apply the recursive ``overrides`` tree implemented by
``canonical_hyperparameter_search_config``.  The historical flat training and
self-contained search schemas are intentionally not supported here.
"""

from __future__ import annotations

from typing import Any, TypeAlias

from sequifier.config.canonical_hyperparameter_search_config import (
    CanonicalHyperparameterSearchConfig,
    compile_canonical_hyperparameter_search_config,
)

PartialHyperparameterSearchConfig: TypeAlias = CanonicalHyperparameterSearchConfig


def compile_hyperparameter_search_override_config(
    config_path: str,
    config_values: dict[str, Any],
    skip_metadata: bool,
) -> CanonicalHyperparameterSearchConfig:
    """Compile canonical base-training plus partial recursive overrides."""

    return compile_canonical_hyperparameter_search_config(
        config_path,
        config_values,
        skip_metadata,
    )


__all__ = [
    "PartialHyperparameterSearchConfig",
    "compile_hyperparameter_search_override_config",
]
