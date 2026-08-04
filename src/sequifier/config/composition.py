"""Composition helpers for user-authored configuration fragments."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from typing import Any

ConfigPath = tuple[str, ...]


# These values are complete, typed components.  Replacing them as a unit keeps
# fields from one discriminator variant from leaking into another variant.
DEFAULT_ATOMIC_PATHS: frozenset[ConfigPath] = frozenset(
    {
        ("model_spec", "ingestion_spec"),
        ("model_spec", "ingestion_merge"),
        ("model_spec", "decoding_spec"),
        ("training_spec", "optimizer"),
        ("training_spec", "scheduler"),
        ("training_spec", "bert_spec", "span_masking"),
    }
)


def deep_merge_config(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
    *,
    atomic_paths: frozenset[ConfigPath] = DEFAULT_ATOMIC_PATHS,
) -> dict[str, Any]:
    """Return a deep merge of two authored mappings.

    Dictionaries merge recursively.  Lists and scalar values are replaced,
    and an explicit ``None`` clears the inherited value.  Neither input is
    mutated.
    """

    return _deep_merge_dicts(base, override, (), atomic_paths)


def merge_config_fragments(
    fragments: Iterable[Mapping[str, Any]],
    *,
    atomic_paths: frozenset[ConfigPath] = DEFAULT_ATOMIC_PATHS,
) -> dict[str, Any]:
    """Merge authored fragments in order, with later fragments taking priority."""

    merged: dict[str, Any] = {}
    for fragment in fragments:
        if not isinstance(fragment, Mapping):
            raise TypeError("Configuration fragments must be mappings")
        merged = _deep_merge_dicts(merged, fragment, (), atomic_paths)
    return merged


def _deep_merge_dicts(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
    path: ConfigPath,
    atomic_paths: frozenset[ConfigPath],
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, override_value in override.items():
        child_path = (*path, str(key))
        base_value = merged.get(key)

        if (
            child_path not in atomic_paths
            and isinstance(base_value, Mapping)
            and isinstance(override_value, Mapping)
            and not _changes_discriminator(base_value, override_value)
        ):
            merged[key] = _deep_merge_dicts(
                base_value,
                override_value,
                child_path,
                atomic_paths,
            )
        else:
            merged[key] = copy.deepcopy(override_value)
    return merged


def _changes_discriminator(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> bool:
    for discriminator in ("type", "name"):
        if (
            discriminator in base
            and discriminator in override
            and base[discriminator] != override[discriminator]
        ):
            return True
    return False
