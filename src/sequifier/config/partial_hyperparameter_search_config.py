"""Compile base-training plus partial-search configs to the legacy HPS model.

The user-facing partial format intentionally keeps the original untagged list and
distribution grammar.  This module owns the small amount of schema knowledge that
is genuinely specific to hyperparameter search: singleton inheritance, atomic
frontend values, coupled candidate groups, and metadata-derived top-level values.
All other fields are discovered from the concrete and legacy Pydantic models.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from sequifier.config.train_config import SequifierConfig, load_train_config_with_source
from sequifier.helpers import normalize_path, stored_window_layout_from_metadata
from sequifier.objectives import (
    BERTObjective,
    NextOccurrenceObjective,
    get_objective_class,
)
from sequifier.special_tokens import SPECIAL_TOKEN_IDS, validate_special_token_ids

from .hyperparameter_search_config import (
    BERTSpecHyperparameterSampling,
    HyperparameterSearchConfig,
    ModelSpecHyperparameterSampling,
    TrainingSpecHyperparameterSampling,
)

ConfigPath = tuple[str, ...]
_MISSING = object()


class PartialHyperparameterSearchConfig(BaseModel):
    """Search controls plus a raw, schema-checked partial training override."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        populate_by_name=True,
    )

    base_config_path: str
    overrides: dict[str, Any]

    hp_search_name: str
    search_strategy: str = "bayesian"
    global_seed: int | None = None
    n_trials: int | None = Field(None, alias="n_samples")
    prune_trials: bool | None = True
    pruning_warmup_epochs: int | None = Field(default=None, ge=0)
    pruning_warmup_batches: int | None = Field(default=None, ge=0)
    model_config_write_path: str

    evaluation_inference_config: str | None = None
    evaluation_script: str | None = None
    evaluation_metric_directions: list[str] | None = None
    evaluation_metrics: list[str] | None = None
    override_input: bool = False

    @model_validator(mode="after")
    def validate_overrides(self):
        _validate_override_mapping(self.overrides)
        return self


class _OverrideMapping(RootModel[dict[str, Any]]):
    """Compatibility base for the formerly handwritten partial model classes."""

    def __init__(self, root: dict[str, Any] | None = None, **values: Any):
        if root is not None and values:
            raise TypeError("pass either a root mapping or keyword override values")
        super().__init__(root={} if root is None and not values else root or values)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in self.root:
                return self.root[name]
            raise


class BERTSpecHyperparameterSamplingOverride(_OverrideMapping):
    """Compatibility wrapper for a partial BERT search mapping."""

    @model_validator(mode="after")
    def validate_mapping(self):
        _validate_override_mapping({"training_spec": {"bert_spec": self.root}})
        return self


class TrainingSpecHyperparameterSamplingOverride(_OverrideMapping):
    """Compatibility wrapper for a partial training search mapping."""

    @model_validator(mode="after")
    def validate_mapping(self):
        _validate_override_mapping({"training_spec": self.root})
        return self


class ModelSpecHyperparameterSamplingOverride(_OverrideMapping):
    """Compatibility wrapper for a partial model search mapping."""

    @model_validator(mode="after")
    def validate_mapping(self):
        _validate_override_mapping({"model_spec": self.root})
        return self


class HyperparameterSearchOverrides(_OverrideMapping):
    """Compatibility wrapper for a top-level partial search mapping."""

    @model_validator(mode="after")
    def validate_mapping(self):
        _validate_override_mapping(self.root)
        return self


@dataclass(frozen=True)
class SearchFieldPolicy:
    """How a concrete base value becomes a legacy search-space value."""

    inheritance: Literal[
        "fixed",
        "singleton",
        "authored_fixed",
        "authored_singleton",
    ] = "fixed"
    singleton_null_override: bool = False


def _policies(
    paths: set[ConfigPath],
    inheritance: Literal[
        "fixed",
        "singleton",
        "authored_fixed",
        "authored_singleton",
    ],
) -> dict[ConfigPath, SearchFieldPolicy]:
    return {path: SearchFieldPolicy(inheritance) for path in paths}


SEARCH_FIELD_POLICIES: dict[ConfigPath, SearchFieldPolicy] = {
    **_policies(
        {
            ("context_length",),
            ("seed",),
            ("model_spec", "dim_model"),
            ("model_spec", "n_head"),
            ("model_spec", "dim_feedforward"),
            ("model_spec", "num_layers"),
            ("model_spec", "decoding_support"),
            ("model_spec", "activation_fn"),
            ("model_spec", "normalization"),
            ("model_spec", "positional_encoding"),
            ("model_spec", "positional_encoding_scope"),
            ("model_spec", "attention_type"),
            ("model_spec", "attention_output_projection"),
            ("model_spec", "norm_first"),
            ("model_spec", "n_kv_heads"),
            ("model_spec", "rope_theta"),
            ("training_spec", "epochs"),
            ("training_spec", "training_objective"),
            ("training_spec", "batch_size"),
            ("training_spec", "learning_rate"),
            ("training_spec", "accumulation_steps"),
            ("training_spec", "gradient_clip"),
            ("training_spec", "dropout"),
            ("training_spec", "optimizer"),
            ("training_spec", "scheduler"),
            ("training_spec", "bert_spec", "masking_probability"),
            ("training_spec", "bert_spec", "replacement_distribution"),
            ("training_spec", "bert_spec", "span_masking"),
        },
        "singleton",
    ),
    **_policies(
        {
            ("model_spec", "ingestion_spec"),
            ("model_spec", "ingestion_merge"),
        },
        "authored_fixed",
    ),
    **_policies(
        {("model_spec", "decoding_spec")},
        "authored_singleton",
    ),
    ("training_spec", "accumulation_steps"): SearchFieldPolicy(
        "singleton",
        singleton_null_override=True,
    ),
}


@dataclass(frozen=True)
class CoupledCandidateGroup:
    """Candidate fields selected together by one legacy trial index."""

    name: str
    paths: tuple[ConfigPath, ...]


COUPLED_CANDIDATE_GROUPS = (
    CoupledCandidateGroup(
        "data schema (input_columns, column_data_types)",
        (("input_columns",), ("column_data_types",)),
    ),
    CoupledCandidateGroup(
        "model width (dim_model, n_head, ingestion_spec, ingestion_merge)",
        (
            ("model_spec", "dim_model"),
            ("model_spec", "n_head"),
            ("model_spec", "ingestion_spec"),
            ("model_spec", "ingestion_merge"),
        ),
    ),
    CoupledCandidateGroup(
        "training schedule (learning_rate, epochs, scheduler)",
        (
            ("training_spec", "learning_rate"),
            ("training_spec", "epochs"),
            ("training_spec", "scheduler"),
        ),
    ),
)

_DATA_GROUP, _MODEL_WIDTH_GROUP, _TRAINING_SCHEDULE_GROUP = COUPLED_CANDIDATE_GROUPS

_TOP_LEVEL_DERIVED_FIELDS = {
    "categorical_columns",
    "model_name",
    "real_columns",
    "window_view",
}
_TOP_LEVEL_OVERRIDE_FIELDS = (
    set(SequifierConfig.model_fields) - _TOP_LEVEL_DERIVED_FIELDS
) | {"context_length", "target_offset"}
_NESTED_OVERRIDE_MODELS: dict[ConfigPath, type[BaseModel]] = {
    ("model_spec",): ModelSpecHyperparameterSampling,
    ("training_spec",): TrainingSpecHyperparameterSampling,
    ("training_spec", "bert_spec"): BERTSpecHyperparameterSampling,
}
_NULL_COMPATIBILITY_EXCEPTIONS = {
    ("input_columns",),
    ("training_spec", "accumulation_steps"),
}
_SKIP_DIRECT_TYPE_VALIDATION = {
    ("training_spec", "optimizer"),
    ("training_spec", "scheduler"),
    ("training_spec", "training_objective"),
}


def _path_text(path: ConfigPath) -> str:
    return ".".join(("overrides", *path))


def _field_for_path(path: ConfigPath):
    if len(path) == 1:
        return HyperparameterSearchConfig.model_fields[path[0]]
    section_model = _NESTED_OVERRIDE_MODELS[path[:-1]]
    return section_model.model_fields[path[-1]]


@lru_cache(maxsize=None)
def _field_adapter(path: ConfigPath) -> TypeAdapter:
    return TypeAdapter(_field_for_path(path).annotation)


def _allows_explicit_null(path: ConfigPath) -> bool:
    if path in _NULL_COMPATIBILITY_EXCEPTIONS:
        return True
    try:
        _field_adapter(path).validate_python(None)
    except ValidationError:
        return False
    return True


def _validate_named_component_candidates(path: ConfigPath, value: Any) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{_path_text(path)} must be a list of named mappings")
    for index, candidate in enumerate(value):
        if not isinstance(candidate, dict):
            raise ValueError(f"{_path_text(path)}.{index} must be a named mapping")
        name = candidate.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{_path_text(path)}.{index} must define a non-empty string 'name'"
            )


def _validate_override_value(path: ConfigPath, value: Any) -> None:
    if value is None:
        if path in {("model_spec",), ("training_spec",)}:
            raise ValueError(
                f"{_path_text(path)}: explicit null is only valid for nullable "
                f"fields; cannot clear {path[-1]!r}"
            )
        if not _allows_explicit_null(path):
            raise ValueError(
                f"{_path_text(path)}: explicit null is only valid for nullable "
                f"fields; cannot clear {path[-1]!r}"
            )
        return

    if path in {
        ("training_spec", "optimizer"),
        ("training_spec", "scheduler"),
    }:
        _validate_named_component_candidates(path, value)
        return
    if path == ("training_spec", "training_objective") and isinstance(value, str):
        return
    if path in _SKIP_DIRECT_TYPE_VALIDATION:
        return

    try:
        _field_adapter(path).validate_python(value)
    except ValidationError as error:
        raise ValueError(f"Invalid {_path_text(path)}:\n{error}") from error


def _validate_override_mapping(
    values: dict[str, Any],
    prefix: ConfigPath = (),
) -> None:
    if prefix:
        allowed_fields = set(_NESTED_OVERRIDE_MODELS[prefix].model_fields)
    else:
        allowed_fields = _TOP_LEVEL_OVERRIDE_FIELDS

    for field_name, value in values.items():
        path = (*prefix, field_name)
        if field_name not in allowed_fields:
            raise ValueError(f"Unknown override field {_path_text(path)!r}")

        nested_prefix = path if path in _NESTED_OVERRIDE_MODELS else None
        if nested_prefix is not None:
            if value is None:
                _validate_override_value(path, value)
            elif not isinstance(value, dict):
                raise ValueError(f"{_path_text(path)} must be a mapping")
            else:
                _validate_override_mapping(value, nested_prefix)
            continue

        _validate_override_value(path, value)


def _get_path(values: dict[str, Any], path: ConfigPath) -> Any:
    current: Any = values
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _is_configured(values: dict[str, Any], path: ConfigPath) -> bool:
    return _get_path(values, path) is not _MISSING


def _configured_or_base(
    overrides: dict[str, Any],
    path: ConfigPath,
    base_value: Any,
) -> Any:
    value = _get_path(overrides, path)
    return copy.deepcopy(base_value if value is _MISSING else value)


def _coupled_candidate_count(
    config_path: str,
    overrides: dict[str, Any],
    group: CoupledCandidateGroup,
) -> int:
    lengths: dict[str, int] = {}
    for path in group.paths:
        value = _get_path(overrides, path)
        if isinstance(value, list):
            lengths[".".join(path)] = len(value)

    empty_fields = [name for name, length in lengths.items() if length == 0]
    if empty_fields:
        raise ValueError(
            f"Override config '{config_path}' configures empty candidate lists "
            f"for coupled group {group.name!r}: {empty_fields}."
        )
    if len(set(lengths.values())) > 1:
        details = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(
            f"Override config '{config_path}' configures incompatible candidate "
            f"counts for coupled group {group.name!r}: {details}. Only "
            "non-overridden members can be repeated automatically."
        )
    return next(iter(lengths.values()), 1)


def _compiled_override_value(path: ConfigPath, value: Any) -> Any:
    policy = SEARCH_FIELD_POLICIES.get(path, SearchFieldPolicy())
    if value is None and policy.singleton_null_override:
        return [None]
    return copy.deepcopy(value)


def _inherited_search_value(
    path: ConfigPath,
    value: Any,
    source_values: dict[str, Any],
) -> Any:
    policy = SEARCH_FIELD_POLICIES.get(path, SearchFieldPolicy())
    if policy.inheritance == "singleton":
        return [copy.deepcopy(value)]
    if policy.inheritance == "authored_fixed":
        return copy.deepcopy(value) if source_values.get(path[-1]) is not None else None
    if policy.inheritance == "authored_singleton":
        return (
            [copy.deepcopy(value)] if source_values.get(path[-1]) is not None else None
        )
    return copy.deepcopy(value)


def _repeat_inherited_group_members(
    compiled_values: dict[str, Any],
    overrides: dict[str, Any],
    group: CoupledCandidateGroup,
    candidate_count: int,
    prefix: ConfigPath,
) -> None:
    if candidate_count == 1:
        return

    for path in group.paths:
        if path[:-1] != prefix or _is_configured(overrides, path):
            continue
        field_name = path[-1]
        value = compiled_values[field_name]
        policy = SEARCH_FIELD_POLICIES.get(path, SearchFieldPolicy())
        if policy.inheritance == "singleton":
            inherited = value[0]
            compiled_values[field_name] = [
                copy.deepcopy(inherited) for _ in range(candidate_count)
            ]
        elif policy.inheritance == "authored_fixed" and value is not None:
            compiled_values[field_name] = [
                copy.deepcopy(value) for _ in range(candidate_count)
            ]


def _compile_bert_sampling(
    base_bert_spec: BaseModel | None,
    source_values: dict[str, Any],
    override_values: dict[str, Any],
) -> dict[str, Any]:
    base_values = (
        {} if base_bert_spec is None else base_bert_spec.model_dump(mode="python")
    )
    compiled = {
        field_name: _inherited_search_value(
            ("training_spec", "bert_spec", field_name),
            value,
            source_values,
        )
        for field_name, value in base_values.items()
        if field_name in BERTSpecHyperparameterSampling.model_fields
    }
    compiled.update(copy.deepcopy(override_values))
    return compiled


def _compile_sampling_section(
    config_path: str,
    base_model: BaseModel,
    source_values: dict[str, Any],
    overrides: dict[str, Any],
    prefix: ConfigPath,
    target_model: type[BaseModel],
    coupled_group: CoupledCandidateGroup,
    inherited_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_values = base_model.model_dump(mode="python")
    if inherited_values is not None:
        base_values.update(copy.deepcopy(inherited_values))
    override_values = _get_path(overrides, prefix)
    if override_values is _MISSING:
        override_values = {}

    compiled: dict[str, Any] = {}
    for field_name in target_model.model_fields:
        if field_name == "bert_spec" or field_name not in base_values:
            continue
        compiled[field_name] = _inherited_search_value(
            (*prefix, field_name),
            base_values[field_name],
            source_values,
        )

    compiled.update(
        {
            field_name: _compiled_override_value((*prefix, field_name), value)
            for field_name, value in override_values.items()
            if field_name != "bert_spec"
        }
    )

    candidate_count = _coupled_candidate_count(
        config_path,
        overrides,
        coupled_group,
    )
    _repeat_inherited_group_members(
        compiled,
        overrides,
        coupled_group,
        candidate_count,
        prefix,
    )

    if prefix != ("training_spec",):
        return compiled

    base_bert_spec = getattr(base_model, "bert_spec")
    bert_override = override_values.get("bert_spec", _MISSING)
    source_bert_spec = source_values.get("bert_spec") or {}
    if bert_override is None:
        compiled["bert_spec"] = None
    elif isinstance(bert_override, dict):
        compiled["bert_spec"] = _compile_bert_sampling(
            base_bert_spec,
            source_bert_spec,
            bert_override,
        )
    elif base_bert_spec is not None:
        compiled["bert_spec"] = _compile_bert_sampling(
            base_bert_spec,
            source_bert_spec,
            {},
        )
    else:
        compiled["bert_spec"] = None

    objective_classes = [
        get_objective_class(objective_name)
        for objective_name in (
            [compiled["training_objective"]]
            if isinstance(compiled["training_objective"], str)
            else compiled["training_objective"]
        )
    ]
    has_bert_objective = any(
        issubclass(objective_class, BERTObjective)
        for objective_class in objective_classes
    )
    has_next_occurrence_objective = any(
        issubclass(objective_class, NextOccurrenceObjective)
        for objective_class in objective_classes
    )
    if not has_bert_objective and bert_override is _MISSING:
        compiled["bert_spec"] = None
    if (
        not has_next_occurrence_objective
        and "next_occurrence_config" not in override_values
    ):
        compiled["next_occurrence_config"] = None

    return compiled


def _resolve_base_config_path(config_path: str, base_config_path: str) -> str:
    if os.path.isabs(base_config_path) or os.path.exists(base_config_path):
        return base_config_path
    return os.path.join(os.path.dirname(os.path.abspath(config_path)), base_config_path)


def compile_hyperparameter_search_override_config(
    config_path: str,
    config_values: dict[str, Any],
    skip_metadata: bool,
) -> HyperparameterSearchConfig:
    """Compile a training config and raw partial override to the legacy model."""
    try:
        partial = PartialHyperparameterSearchConfig.model_validate(config_values)
    except (ValidationError, ValueError) as error:
        raise ValueError(
            f"Invalid override hyperparameter search config '{config_path}':\n{error}"
        ) from error

    base_config_path = _resolve_base_config_path(
        config_path,
        partial.base_config_path,
    )
    try:
        loaded_base = load_train_config_with_source(
            base_config_path,
            {},
            skip_metadata,
        )
    except Exception as error:
        raise ValueError(
            f"Unable to load base training config '{base_config_path}' referenced "
            f"by override config '{config_path}': {error}"
        ) from error

    base_model = loaded_base.model
    source_values = loaded_base.source_values
    overrides = partial.overrides

    project_root = _configured_or_base(
        overrides,
        ("project_root",),
        base_model.project_root,
    )
    metadata_config_path = _configured_or_base(
        overrides,
        ("metadata_config_path",),
        base_model.metadata_config_path,
    )

    metadata_changed = (
        project_root != base_model.project_root
        or metadata_config_path != base_model.metadata_config_path
    )
    metadata_config = (
        None if skip_metadata else copy.deepcopy(loaded_base.metadata_values)
    )
    if not skip_metadata and metadata_changed:
        effective_metadata_path = normalize_path(
            metadata_config_path,
            project_root,
        )
        try:
            with open(effective_metadata_path, "r") as file:
                metadata_config = json.loads(file.read())
        except Exception as error:
            raise ValueError(
                f"Override config '{config_path}' changes project_root or "
                f"metadata_config_path, but metadata could not be loaded from "
                f"'{effective_metadata_path}': {error}"
            ) from error

    if metadata_config is not None:
        storage_layout = stored_window_layout_from_metadata(metadata_config)
        if storage_layout.version != 2:
            raise ValueError(
                f"Override config '{config_path}' requires metadata "
                "stored_window_layout_version=2, got "
                f"{storage_layout.version}."
            )
        base_column_types = (
            source_values.get("column_data_types")
            or metadata_config["column_data_types"]
        )
        base_n_classes = source_values.get("n_classes") or metadata_config["n_classes"]
        base_id_maps = metadata_config["id_maps"]
        base_special_token_ids = validate_special_token_ids(
            metadata_config.get(
                "special_token_ids",
                SPECIAL_TOKEN_IDS.ids_by_label,
            ),
            source=f"metadata config '{metadata_config_path}'",
        )
        split_paths = metadata_config["split_paths"]
        raw_training_path = source_values.get("data_path") or split_paths[0]
        raw_validation_path = (
            source_values.get("validation_data_path")
            or split_paths[min(1, len(split_paths) - 1)]
        )
        base_training_path = normalize_path(raw_training_path, project_root)
        base_validation_path = normalize_path(raw_validation_path, project_root)
        source_input_columns = source_values.get("input_columns")
        base_input_columns = (
            list(base_column_types)
            if source_input_columns is None
            else source_input_columns
        )
    else:
        storage_layout = base_model.storage_layout
        base_column_types = base_model.column_data_types
        base_n_classes = base_model.n_classes
        base_id_maps = base_model.id_maps
        base_special_token_ids = base_model.special_token_ids
        base_training_path = base_model.data_path
        base_validation_path = base_model.validation_data_path
        base_input_columns = base_model.input_columns

    data_path = _configured_or_base(
        overrides,
        ("data_path",),
        base_training_path,
    )
    validation_data_path = _configured_or_base(
        overrides,
        ("validation_data_path",),
        base_validation_path,
    )
    if _is_configured(overrides, ("data_path",)):
        data_path = normalize_path(data_path, project_root)
    if _is_configured(overrides, ("validation_data_path",)):
        validation_data_path = normalize_path(validation_data_path, project_root)

    input_override = _get_path(overrides, ("input_columns",))
    column_override = _get_path(overrides, ("column_data_types",))
    data_candidate_count = _coupled_candidate_count(
        config_path,
        overrides,
        _DATA_GROUP,
    )
    if column_override is not _MISSING:
        column_data_types = copy.deepcopy(column_override)
    else:
        column_data_types = [
            copy.deepcopy(base_column_types) for _ in range(data_candidate_count)
        ]

    if input_override is None:
        input_columns = [list(candidate) for candidate in column_data_types]
    elif input_override is not _MISSING:
        input_columns = copy.deepcopy(input_override)
    else:
        input_columns = [
            copy.deepcopy(base_input_columns) for _ in range(data_candidate_count)
        ]

    categorical_columns = [
        [
            column
            for column in input_candidate
            if "int" in column_candidate.get(column, "").lower()
        ]
        for input_candidate, column_candidate in zip(input_columns, column_data_types)
    ]
    real_columns = [
        [
            column
            for column in input_candidate
            if "float" in column_candidate.get(column, "").lower()
        ]
        for input_candidate, column_candidate in zip(input_columns, column_data_types)
    ]

    source_model_spec = source_values.get("model_spec", {})
    source_training_spec = copy.deepcopy(source_values.get("training_spec", {}))
    source_training_spec.update(
        {
            "training_objective": source_values.get(
                "training_objective", base_model.training_objective
            ),
            "device": source_values.get("device", base_model.device),
        }
    )
    source_window_view = source_values.get("window_view", {})
    inherited_target_offset = source_values.get(
        "target_offset",
        source_window_view.get("target_offset", 1),
    )
    model_sampling = _compile_sampling_section(
        config_path,
        base_model.model_spec,
        source_model_spec,
        overrides,
        ("model_spec",),
        ModelSpecHyperparameterSampling,
        _MODEL_WIDTH_GROUP,
    )
    training_sampling = _compile_sampling_section(
        config_path,
        base_model.training_spec,
        source_training_spec,
        overrides,
        ("training_spec",),
        TrainingSpecHyperparameterSampling,
        _TRAINING_SCHEDULE_GROUP,
        inherited_values={
            "training_objective": base_model.training_objective,
            "device": base_model.device,
        },
    )

    search_values = partial.model_dump(
        exclude={"base_config_path", "overrides"},
        mode="python",
        by_alias=True,
    )
    base_top_values = base_model.model_dump(mode="python")
    base_top_values.update(
        {
            "project_root": project_root,
            "metadata_config_path": metadata_config_path,
            "data_path": data_path,
            "validation_data_path": validation_data_path,
            "input_columns": input_columns,
            "column_data_types": column_data_types,
            "categorical_columns": categorical_columns,
            "real_columns": real_columns,
            "id_maps": base_id_maps,
            "special_token_ids": base_special_token_ids,
            "context_length": base_model.window_view.context_length,
            "target_offset": inherited_target_offset,
            "storage_layout": storage_layout,
            "n_classes": base_n_classes,
        }
    )

    compiled_values: dict[str, Any] = dict(search_values)
    special_top_fields = {
        "categorical_columns",
        "column_data_types",
        "input_columns",
        "model_hyperparameter_sampling",
        "real_columns",
        "training_hyperparameter_sampling",
        "data_path",
        "validation_data_path",
    }
    for field_name in HyperparameterSearchConfig.model_fields:
        if field_name in compiled_values or field_name in special_top_fields:
            continue
        if field_name not in base_top_values:
            continue
        override_value = _get_path(overrides, (field_name,))
        if override_value is not _MISSING:
            compiled_values[field_name] = copy.deepcopy(override_value)
        else:
            compiled_values[field_name] = _inherited_search_value(
                (field_name,),
                base_top_values[field_name],
                source_values,
            )

    compiled_values.update(
        {
            "data_path": data_path,
            "validation_data_path": validation_data_path,
            "input_columns": input_columns,
            "column_data_types": column_data_types,
            "categorical_columns": categorical_columns,
            "real_columns": real_columns,
            "model_hyperparameter_sampling": model_sampling,
            "training_hyperparameter_sampling": training_sampling,
        }
    )

    try:
        return HyperparameterSearchConfig.model_validate(compiled_values)
    except ValidationError as error:
        raise ValueError(
            f"Failed to compile override hyperparameter search config "
            f"'{config_path}' with base training config '{base_config_path}':\n{error}"
        ) from error
