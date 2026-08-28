"""Composition helpers for user-authored configuration fragments."""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable, Mapping
from typing import Any

import yaml

from sequifier.typechecking import beartype

ConfigPath = tuple[str, ...]

ADDITIONAL_CONFIG_PATHS_KEY = "additional_config_paths"


# These values are complete, typed components.  Replacing them as a unit keeps
# fields from one discriminator variant from leaking into another variant.
DEFAULT_ATOMIC_PATHS: frozenset[ConfigPath] = frozenset(
    {
        ("model", "backbone"),
        ("global_training", "optimizer"),
        ("global_training", "scheduler"),
        ("global_training", "bert_spec", "span_masking"),
    }
)


@beartype
def _is_atomic(path: ConfigPath, atomic_paths: frozenset[ConfigPath]) -> bool:
    if path in atomic_paths:
        return True
    named_interface_component = len(path) == 4 and path[:2] == (
        "model",
        "interfaces",
    )
    singleton_interface_component = len(path) == 3 and path[:2] == (
        "model",
        "interface",
    )
    return (named_interface_component or singleton_interface_component) and path[
        -1
    ] in {"ingestion", "decoder"}


@beartype
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


@beartype
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


@beartype
def load_composed_yaml_config(
    config_path: str,
    *,
    atomic_paths: frozenset[ConfigPath] = DEFAULT_ATOMIC_PATHS,
) -> dict[str, Any]:
    """Load one YAML config and its direct, complementary fragments.

    Relative fragment paths are resolved against the entry config's
    ``project_root``.  Fragments cannot include further fragments.  Duplicate
    authored fields are rejected before the resulting mapping reaches the
    command-specific Pydantic model.
    """

    entry_path = os.path.abspath(config_path)
    entry_values = _load_yaml_mapping(entry_path)
    raw_additional_paths = entry_values.pop(ADDITIONAL_CONFIG_PATHS_KEY, None)
    additional_paths = _normalize_additional_config_paths(
        raw_additional_paths,
        entry_path,
    )

    if not additional_paths:
        return entry_values

    project_root = entry_values.get("project_root")
    if not isinstance(project_root, str) or not project_root.strip():
        raise ValueError(
            f"Config '{entry_path}' must define a non-empty string project_root "
            f"when {ADDITIONAL_CONFIG_PATHS_KEY} is configured."
        )

    fragments: list[tuple[str, Mapping[str, Any]]] = []
    seen_paths = {os.path.realpath(entry_path)}
    for additional_path in additional_paths:
        resolved_path = _resolve_additional_config_path(
            additional_path,
            project_root,
        )
        canonical_path = os.path.realpath(resolved_path)
        if canonical_path in seen_paths:
            raise ValueError(
                f"Config '{entry_path}' references the same configuration file "
                f"more than once: '{resolved_path}'."
            )
        seen_paths.add(canonical_path)

        fragment_values = _load_yaml_mapping(resolved_path)
        if ADDITIONAL_CONFIG_PATHS_KEY in fragment_values:
            raise ValueError(
                f"Config fragment '{resolved_path}' cannot define "
                f"'{ADDITIONAL_CONFIG_PATHS_KEY}'; recursive composition is not "
                "supported."
            )
        fragments.append((resolved_path, fragment_values))

    fragments.append((entry_path, entry_values))
    return merge_complementary_config_fragments(
        fragments,
        atomic_paths=atomic_paths,
    )


@beartype
def merge_complementary_config_fragments(
    fragments: Iterable[tuple[str, Mapping[str, Any]]],
    *,
    atomic_paths: frozenset[ConfigPath] = DEFAULT_ATOMIC_PATHS,
) -> dict[str, Any]:
    """Merge sourced fragments while rejecting duplicate authored fields."""

    merged: dict[str, Any] = {}
    field_sources: dict[ConfigPath, str] = {}
    for source, fragment in fragments:
        if not isinstance(fragment, Mapping):
            raise TypeError("Configuration fragments must be mappings")
        _merge_complementary_dicts(
            merged,
            fragment,
            (),
            source,
            field_sources,
            atomic_paths,
        )
    return merged


@beartype
def _load_yaml_mapping(path: str) -> dict[str, Any]:
    try:
        with open(path, "r") as file:
            values = yaml.safe_load(file)
    except OSError as error:
        raise ValueError(f"Unable to read config '{path}': {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Unable to parse config '{path}': {error}") from error

    if not isinstance(values, dict):
        raise ValueError(f"Config '{path}' must contain a YAML mapping.")
    return values


@beartype
def _normalize_additional_config_paths(
    value: Any,
    config_path: str,
) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        paths = [value]
    elif isinstance(value, list):
        paths = value
    else:
        raise ValueError(
            f"Config '{config_path}' field '{ADDITIONAL_CONFIG_PATHS_KEY}' must "
            "be a non-empty string, a list of non-empty strings, or null."
        )

    if any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ValueError(
            f"Config '{config_path}' field '{ADDITIONAL_CONFIG_PATHS_KEY}' must "
            "be a non-empty string, a list of non-empty strings, or null."
        )
    return paths


@beartype
def _resolve_additional_config_path(path: str, project_root: str) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(project_root, path))


@beartype
def _merge_complementary_dicts(
    merged: dict[str, Any],
    fragment: Mapping[str, Any],
    path: ConfigPath,
    source: str,
    field_sources: dict[ConfigPath, str],
    atomic_paths: frozenset[ConfigPath],
) -> None:
    for key, incoming_value in fragment.items():
        child_path = (*path, str(key))
        if key not in merged:
            merged[key] = copy.deepcopy(incoming_value)
            _record_field_sources(
                incoming_value,
                child_path,
                source,
                field_sources,
                atomic_paths,
            )
            continue

        current_value = merged[key]
        if (
            not _is_atomic(child_path, atomic_paths)
            and isinstance(current_value, Mapping)
            and current_value
            and isinstance(incoming_value, Mapping)
            and incoming_value
        ):
            _merge_complementary_dicts(
                current_value,  # type: ignore[arg-type]
                incoming_value,
                child_path,
                source,
                field_sources,
                atomic_paths,
            )
            continue

        first_source = _field_source_for_path(field_sources, child_path)
        dotted_path = ".".join(child_path)
        raise ValueError(
            f"Duplicate configuration field '{dotted_path}': first defined in "
            f"'{first_source}', also defined in '{source}'."
        )


@beartype
def _record_field_sources(
    value: Any,
    path: ConfigPath,
    source: str,
    field_sources: dict[ConfigPath, str],
    atomic_paths: frozenset[ConfigPath],
) -> None:
    if _is_atomic(path, atomic_paths) or not isinstance(value, Mapping) or not value:
        field_sources[path] = source
        return
    for key, child_value in value.items():
        _record_field_sources(
            child_value,
            (*path, str(key)),
            source,
            field_sources,
            atomic_paths,
        )


@beartype
def _field_source_for_path(
    field_sources: Mapping[ConfigPath, str],
    path: ConfigPath,
) -> str:
    if path in field_sources:
        return field_sources[path]
    for field_path, source in field_sources.items():
        if field_path[: len(path)] == path:
            return source
    return "unknown source"


@beartype
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
            not _is_atomic(child_path, atomic_paths)
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


@beartype
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
