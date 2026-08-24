"""Hyperparameter sampling over canonical authored training configs.

The canonical training schema contains user-named interfaces, datasets, parts,
and phases.  A second hand-maintained mirror of that schema would inevitably
exclude valid training configurations, so this module samples a recursive
override tree and validates every materialized trial with ``SequifierConfig``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import warnings
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    model_validator,
)

from sequifier.config.composable_train_config import (
    SequifierConfig,
    load_train_config_with_source,
)
from sequifier.config.composition import load_composed_yaml_config

ConfigPath = tuple[str | int, ...]
_MISSING = object()
_DISTRIBUTION_KEYS = frozenset({"low", "high", "step", "log", "type"})
_PRIMITIVE_CATEGORICAL_TYPES = (str, int, float, bool, type(None))


def _path_name(path: ConfigPath) -> str:
    """Return a stable, readable Optuna parameter name."""

    value = ""
    for component in path:
        if isinstance(component, int):
            value += f"[{component}]"
        else:
            value += ("." if value else "") + component
    return value or "config"


def _candidate_identity(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            default=str,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return repr(value)


@dataclass(frozen=True)
class _SearchSpace:
    path: ConfigPath
    kind: Literal["categorical", "int", "float"]
    choices: tuple[Any, ...] = ()
    low: int | float | None = None
    high: int | float | None = None
    step: int | float | None = None
    log: bool = False

    @property
    def name(self) -> str:
        return _path_name(self.path)

    def baseline(self) -> Any:
        if self.kind == "categorical":
            return copy.deepcopy(self.choices[0])
        return self.low

    def sample(self, trial: Any) -> Any:
        if self.kind == "int":
            assert isinstance(self.low, int) and isinstance(self.high, int)
            assert isinstance(self.step, int)
            return trial.suggest_int(
                self.name,
                self.low,
                self.high,
                step=self.step,
                log=self.log,
            )
        if self.kind == "float":
            assert self.low is not None and self.high is not None
            return trial.suggest_float(
                self.name,
                float(self.low),
                float(self.high),
                step=None if self.step is None else float(self.step),
                log=self.log,
            )

        if all(
            isinstance(value, _PRIMITIVE_CATEGORICAL_TYPES) for value in self.choices
        ):
            selected = trial.suggest_categorical(self.name, list(self.choices))
            return copy.deepcopy(selected)
        index = trial.suggest_categorical(
            f"{self.name}.__choice_index",
            list(range(len(self.choices))),
        )
        return copy.deepcopy(self.choices[index])

    def grid_size(self) -> int:
        if self.kind == "categorical":
            return len(self.choices)
        if self.kind == "int":
            assert isinstance(self.low, int) and isinstance(self.high, int)
            assert isinstance(self.step, int)
            return (self.high - self.low) // self.step + 1
        if self.step is None:
            raise ValueError(
                f"{self.name}.step must be configured for grid search because "
                "an unstepped float distribution has infinitely many combinations."
            )
        assert self.low is not None and self.high is not None
        low = Decimal(str(self.low))
        high = Decimal(str(self.high))
        step = Decimal(str(self.step))
        return int((high - low) // step) + 1

    def validation_parameters(self) -> list[tuple[str, Any]]:
        """Return trial parameters covering categorical values or range extrema."""

        if self.kind == "categorical":
            if all(
                isinstance(value, _PRIMITIVE_CATEGORICAL_TYPES)
                for value in self.choices
            ):
                return [(self.name, value) for value in self.choices]
            return [
                (f"{self.name}.__choice_index", index)
                for index in range(len(self.choices))
            ]
        if self.kind == "int":
            assert isinstance(self.low, int) and isinstance(self.high, int)
            assert isinstance(self.step, int)
            sampled_high = self.low + ((self.high - self.low) // self.step) * self.step
            values = [self.low]
            if sampled_high != self.low:
                values.append(sampled_high)
            return [(self.name, value) for value in values]

        assert self.low is not None and self.high is not None
        sampled_high = float(self.high)
        if self.step is not None:
            low = Decimal(str(self.low))
            high = Decimal(str(self.high))
            step = Decimal(str(self.step))
            sampled_high = float(low + ((high - low) // step) * step)
        values = [float(self.low)]
        if sampled_high != float(self.low):
            values.append(sampled_high)
        return [(self.name, value) for value in values]


class _ValidationTrial:
    """Minimal Optuna-trial interface used for eager candidate validation."""

    def __init__(self, params: dict[str, Any]):
        self.params = params

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        return self.params.get(name, choices[0])

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int:
        return self.params.get(name, low)

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        return self.params.get(name, low)


class _CompiledValue:
    def materialize(self, trial: Any | None) -> Any:
        raise NotImplementedError

    def spaces(self) -> list[_SearchSpace]:
        return []


@dataclass(frozen=True)
class _LiteralValue(_CompiledValue):
    value: Any

    def materialize(self, trial: Any | None) -> Any:
        return copy.deepcopy(self.value)


@dataclass(frozen=True)
class _SampledValue(_CompiledValue):
    space: _SearchSpace

    def materialize(self, trial: Any | None) -> Any:
        return self.space.baseline() if trial is None else self.space.sample(trial)

    def spaces(self) -> list[_SearchSpace]:
        return [self.space]


@dataclass(frozen=True)
class _MappingValue(_CompiledValue):
    base: dict[str, Any]
    children: dict[str, _CompiledValue]

    def materialize(self, trial: Any | None) -> dict[str, Any]:
        result = copy.deepcopy(self.base)
        for key, child in self.children.items():
            result[key] = child.materialize(trial)
        return result

    def spaces(self) -> list[_SearchSpace]:
        return [space for child in self.children.values() for space in child.spaces()]


def _merge_fixed_patch(base: Any, patch: Any) -> Any:
    """Merge one fixed partial variant using canonical component semantics."""

    if not isinstance(base, dict) or not isinstance(patch, dict):
        return copy.deepcopy(patch)

    result = copy.deepcopy(base)
    if (
        "type" in patch
        and not isinstance(patch["type"], (dict, list))
        and patch["type"] != base.get("type")
    ):
        result = {}
    for key, value in patch.items():
        result[str(key)] = _merge_fixed_patch(result.get(str(key), _MISSING), value)
    return result


def _materialize_against(
    compiled: _CompiledValue,
    current: Any,
    trial: Any | None,
) -> Any:
    """Materialize a compiled override against a dynamically selected variant."""

    if isinstance(compiled, _MappingValue):
        result = (
            copy.deepcopy(current)
            if isinstance(current, dict)
            else copy.deepcopy(compiled.base)
        )
        for key, child in compiled.children.items():
            result[key] = _materialize_against(
                child,
                result.get(key, _MISSING),
                trial,
            )
        return result
    if isinstance(compiled, _ListValue):
        result = (
            copy.deepcopy(current)
            if isinstance(current, list)
            else copy.deepcopy(compiled.base)
        )
        for index, child in compiled.children.items():
            result[index] = _materialize_against(child, result[index], trial)
        return result
    return compiled.materialize(trial)


@dataclass(frozen=True)
class _VariantMappingValue(_CompiledValue):
    """A partial paired variant followed by independent sibling overrides."""

    base: dict[str, Any]
    variants: _SampledValue
    children: dict[str, _CompiledValue]

    def materialize(self, trial: Any | None) -> dict[str, Any]:
        variant = self.variants.materialize(trial)
        result = _merge_fixed_patch(self.base, variant)
        for key, child in self.children.items():
            result[key] = _materialize_against(
                child,
                result.get(key, _MISSING),
                trial,
            )
        return result

    def spaces(self) -> list[_SearchSpace]:
        return [
            *self.variants.spaces(),
            *(space for child in self.children.values() for space in child.spaces()),
        ]


@dataclass(frozen=True)
class _ListValue(_CompiledValue):
    base: list[Any]
    children: dict[int, _CompiledValue]

    def materialize(self, trial: Any | None) -> list[Any]:
        result = copy.deepcopy(self.base)
        for index, child in self.children.items():
            result[index] = child.materialize(trial)
        return result

    def spaces(self) -> list[_SearchSpace]:
        return [space for child in self.children.values() for space in child.spaces()]


def _categorical_space(path: ConfigPath, choices: Any) -> _SampledValue:
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"{_path_name(path)} choices must be a non-empty list")
    identities = [_candidate_identity(value) for value in choices]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{_path_name(path)} choices cannot contain duplicates")
    return _SampledValue(
        _SearchSpace(path=path, kind="categorical", choices=tuple(choices))
    )


def _distribution_space(
    path: ConfigPath,
    expression: dict[str, Any],
    base: Any,
) -> _SampledValue:
    low = expression["low"]
    high = expression["high"]
    step = expression.get("step")
    log = expression.get("log", False)
    explicit_type = expression.get("type")

    if isinstance(low, bool) or isinstance(high, bool):
        raise ValueError(f"{_path_name(path)} distribution bounds must be numeric")
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise ValueError(f"{_path_name(path)} distribution bounds must be numeric")
    if low > high:
        raise ValueError(
            f"{_path_name(path)} distribution low must be <= high, got {low} > {high}"
        )
    if explicit_type not in {None, "int", "float"}:
        raise ValueError(
            f"{_path_name(path)} distribution type must be 'int' or 'float'"
        )
    if not isinstance(log, bool):
        raise ValueError(f"{_path_name(path)} distribution log must be a boolean")
    if log and low <= 0:
        raise ValueError(f"{_path_name(path)} log distributions require low > 0")

    is_int = explicit_type == "int" or (
        explicit_type is None
        and isinstance(base, int)
        and not isinstance(base, bool)
        and isinstance(low, int)
        and isinstance(high, int)
    )
    if explicit_type is None and (base is _MISSING or base is None):
        is_int = isinstance(low, int) and isinstance(high, int)

    if is_int:
        if not isinstance(low, int) or not isinstance(high, int):
            raise ValueError(
                f"{_path_name(path)} integer distribution requires integer bounds"
            )
        if step is None:
            step = 1
        if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
            raise ValueError(
                f"{_path_name(path)} integer distribution step must be a "
                "positive integer"
            )
        if log and step != 1:
            raise ValueError(
                f"{_path_name(path)} log integer distributions require step=1"
            )
        return _SampledValue(
            _SearchSpace(
                path=path,
                kind="int",
                low=low,
                high=high,
                step=step,
                log=log,
            )
        )

    if step is not None and (
        not isinstance(step, (int, float)) or isinstance(step, bool) or step <= 0
    ):
        raise ValueError(f"{_path_name(path)} float distribution step must be positive")
    if log and step is not None:
        raise ValueError(
            f"{_path_name(path)} log float distributions cannot configure step"
        )
    return _SampledValue(
        _SearchSpace(
            path=path,
            kind="float",
            low=float(low),
            high=float(high),
            step=None if step is None else float(step),
            log=log,
        )
    )


def _is_distribution(expression: dict[str, Any]) -> bool:
    keys = set(expression)
    return {"low", "high"} <= keys and keys <= _DISTRIBUTION_KEYS


def _list_index_mapping(expression: dict[str, Any]) -> dict[int, Any] | None:
    if not expression:
        return {}
    converted: dict[int, Any] = {}
    for raw_index, value in expression.items():
        if isinstance(raw_index, int):
            index = raw_index
        elif isinstance(raw_index, str) and raw_index.isdigit():
            index = int(raw_index)
        else:
            return None
        converted[index] = value
    return converted


def _compile_value(expression: Any, base: Any, path: ConfigPath) -> _CompiledValue:
    if (
        isinstance(expression, dict)
        and set(expression) in ({"choices"}, {"$choices"})
        and not (isinstance(base, dict) and next(iter(expression)) in base)
    ):
        key = "choices" if "choices" in expression else "$choices"
        return _categorical_space(path, expression[key])
    if (
        isinstance(expression, dict)
        and set(expression) in ({"fixed"}, {"$fixed"})
        and not (isinstance(base, dict) and next(iter(expression)) in base)
    ):
        key = "fixed" if "fixed" in expression else "$fixed"
        return _LiteralValue(expression[key])
    if (
        isinstance(expression, dict)
        and not isinstance(base, dict)
        and _is_distribution(expression)
    ):
        return _distribution_space(path, expression, base)

    if isinstance(base, list) and isinstance(expression, dict):
        indexed = _list_index_mapping(expression)
        if indexed is not None:
            children: dict[int, _CompiledValue] = {}
            for index, child_expression in indexed.items():
                if index < 0 or index >= len(base):
                    raise ValueError(
                        f"{_path_name(path)} index {index} is outside the base list"
                    )
                children[index] = _compile_value(
                    child_expression,
                    base[index],
                    (*path, index),
                )
            return _ListValue(copy.deepcopy(base), children)

    if isinstance(expression, dict):
        base_mapping = copy.deepcopy(base) if isinstance(base, dict) else {}
        # Discriminated component variants are complete types.  Changing the
        # discriminator starts from an empty mapping so fields from the old
        # variant cannot leak into the sampled value.
        if (
            isinstance(base, dict)
            and "type" in expression
            and not isinstance(expression["type"], (dict, list))
            and expression["type"] != base.get("type")
        ):
            base_mapping = {}
        variant_keys = {"variants", "$variants"} & set(expression)
        if len(variant_keys) > 1:
            raise ValueError(
                f"{_path_name(path)} cannot configure both variants and $variants"
            )
        variant_key = next(iter(variant_keys), None)
        child_expressions = {
            key: value for key, value in expression.items() if key != variant_key
        }
        mapping_children = {
            str(key): _compile_value(
                child_expression,
                base_mapping.get(str(key), _MISSING),
                (*path, str(key)),
            )
            for key, child_expression in child_expressions.items()
        }
        if variant_key is not None:
            if not isinstance(base, dict):
                raise ValueError(
                    f"{_path_name(path)} variants require a mapping-valued base"
                )
            variants = expression[variant_key]
            if not isinstance(variants, list) or not all(
                isinstance(variant, dict) for variant in variants
            ):
                raise ValueError(
                    f"{_path_name(path)}.{variant_key} must be a non-empty list "
                    "of partial mappings"
                )
            return _VariantMappingValue(
                base_mapping,
                _categorical_space((*path, "__variant"), variants),
                mapping_children,
            )
        return _MappingValue(base_mapping, mapping_children)

    if isinstance(expression, list):
        if base is _MISSING:
            return _LiteralValue(expression)
        if isinstance(base, list):
            # A list of lists is the established shorthand for selecting a
            # complete list-valued field (for example input_columns).  Other
            # list-valued overrides are fixed replacements; ``choices`` is the
            # unambiguous form for sampling arbitrary list values.
            if expression and all(isinstance(value, list) for value in expression):
                return _categorical_space(path, expression)
            return _LiteralValue(expression)
        return _categorical_space(path, expression)

    return _LiteralValue(expression)


class CanonicalHyperparameterSearchConfig(BaseModel):
    """Search controls and a recursive sampler over one canonical train config."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        populate_by_name=True,
    )

    base_config_path: str = Field(min_length=1)
    overrides: dict[str, Any]
    project_root: str = Field(min_length=1)

    hp_search_name: str = Field(min_length=1)
    search_strategy: Literal["bayesian", "sample", "grid"] = "bayesian"
    global_seed: int | None = None
    n_trials: int | None = Field(default=None, alias="n_samples", gt=0)
    prune_trials: bool = True
    pruning_warmup_epochs: int | None = Field(default=None, ge=0)
    pruning_warmup_batches: int | None = Field(default=None, ge=0)
    model_config_write_path: str = Field(min_length=1)

    evaluation_inference_config: str | None = None
    evaluation_script: str | None = None
    evaluation_metric_directions: list[Literal["minimize", "maximize"]] | None = None
    evaluation_metrics: list[str] | None = None
    _compiled_config: _CompiledValue = PrivateAttr()

    @model_validator(mode="after")
    def validate_search_controls(self):
        reserved_overrides = {"model_name", "project_root"} & set(self.overrides)
        if reserved_overrides:
            raise ValueError(
                "Canonical hyperparameter overrides cannot set run-controlled "
                f"fields: {sorted(reserved_overrides)}"
            )
        if (
            self.pruning_warmup_epochs is not None
            and self.pruning_warmup_batches is not None
        ):
            raise ValueError(
                "Only one of pruning_warmup_epochs and pruning_warmup_batches "
                "can be set."
            )
        if self.evaluation_metrics is not None:
            if not self.evaluation_metrics:
                raise ValueError("evaluation_metrics cannot be empty")
            if self.evaluation_script is None:
                raise ValueError(
                    "evaluation_script must be provided if evaluation_metrics "
                    "is defined."
                )
            if self.evaluation_metric_directions is None:
                raise ValueError(
                    "evaluation_metric_directions must be provided if "
                    "evaluation_metrics is defined."
                )
            if len(self.evaluation_metrics) != len(self.evaluation_metric_directions):
                raise ValueError(
                    "evaluation_metrics and evaluation_metric_directions must have "
                    "the same number of values"
                )
            if self.evaluation_inference_config is None:
                warnings.warn(
                    "Please provide evaluation_inference_config if your "
                    "evaluation_script requires inference outputs",
                    stacklevel=2,
                )
        if self.evaluation_script is not None and not os.path.exists(
            os.path.join(self.project_root, self.evaluation_script)
        ):
            raise ValueError(
                f"evaluation_script {self.evaluation_script!r} does not exist "
                f"under project_root {self.project_root!r}"
            )
        if self.evaluation_inference_config is not None:
            inference_path = self.evaluation_inference_config
            if not os.path.isabs(inference_path):
                inference_path = os.path.join(self.project_root, inference_path)
            if not os.path.exists(inference_path):
                raise ValueError(
                    f"evaluation_inference_config {self.evaluation_inference_config!r} "
                    "does not exist"
                )
        return self

    @property
    def search_spaces(self) -> tuple[_SearchSpace, ...]:
        return tuple(self._compiled_config.spaces())

    def grid_size(self) -> int:
        return math.prod(space.grid_size() for space in self.search_spaces)

    def validate_compiled_search(self) -> None:
        """Validate the representative candidate and finite-grid controls."""

        spaces = self.search_spaces
        baseline_params = {
            name: value
            for space in spaces
            for name, value in space.validation_parameters()[:1]
        }
        candidates: list[tuple[str, _ValidationTrial | None]] = [
            ("the baseline candidate", None)
        ]
        for space in spaces:
            for parameter_name, value in space.validation_parameters():
                parameters = dict(baseline_params)
                parameters[parameter_name] = value
                candidates.append(
                    (
                        f"{space.name}={value!r}",
                        _ValidationTrial(parameters),
                    )
                )

        for description, trial in candidates:
            try:
                SequifierConfig.model_validate(self._materialized_values(trial, 0))
            except ValidationError as error:
                raise ValueError(
                    "Canonical hyperparameter overrides produce an invalid "
                    f"training config for {description}:\n{error}"
                ) from error

        if self.search_strategy == "grid":
            combinations = self.grid_size()
            if self.n_trials is not None and self.n_trials != combinations:
                raise ValueError(
                    "For search_strategy='grid', n_samples must equal the number "
                    f"of configured combinations ({combinations}), got "
                    f"{self.n_trials}. Remove n_samples to run the complete grid."
                )

    def _materialized_values(self, trial: Any | None, run_index: int) -> dict[str, Any]:
        values = self._compiled_config.materialize(trial)
        if not isinstance(values, dict):
            raise ValueError(
                "Canonical hyperparameter overrides must produce a mapping"
            )
        model_override = self.overrides.get("model_spec")
        if isinstance(model_override, dict):
            backbone_override = model_override.get("backbone")
            architecture_override = (
                backbone_override.get("architecture")
                if isinstance(backbone_override, dict)
                else None
            )
            if (
                isinstance(architecture_override, dict)
                and "position_encoding" in architecture_override
                and "positional_encoding_scope" not in architecture_override
            ):
                architecture = values["model_spec"]["backbone"]["architecture"]
                if architecture["position_encoding"]["type"] in {
                    "range",
                    "range_concat",
                }:
                    architecture["positional_encoding_scope"] = "global"
        values["project_root"] = self.project_root
        values["model_name"] = f"{self.hp_search_name}-run-{run_index}"
        return values

    def sample_trial(self, trial: Any, run_index: int) -> SequifierConfig:
        """Sample and validate one concrete canonical authored training config."""

        values = self._materialized_values(trial, run_index)
        try:
            return SequifierConfig.model_validate(values)
        except ValidationError as error:
            parameters = getattr(trial, "params", {})
            raise ValueError(
                "Sampled hyperparameters produce an invalid canonical training "
                f"config for parameters {parameters!r}:\n{error}"
            ) from error


def resolve_base_config_path(config_path: str, base_config_path: str) -> str:
    """Resolve a base config relative to the hyperparameter-search entry file."""

    if os.path.isabs(base_config_path):
        return os.path.abspath(base_config_path)
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(config_path)), base_config_path)
    )


def compile_canonical_hyperparameter_search_config(
    config_path: str,
    config_values: dict[str, Any],
    skip_metadata: bool,
) -> CanonicalHyperparameterSearchConfig:
    """Compile base-training plus recursive overrides into a canonical sampler."""

    base_config_path = resolve_base_config_path(
        config_path,
        config_values["base_config_path"],
    )
    try:
        if skip_metadata:
            base_config = SequifierConfig.model_validate(
                load_composed_yaml_config(base_config_path)
            )
        else:
            base_config = load_train_config_with_source(
                base_config_path,
                {},
                False,
            ).config
    except ValidationError:
        raise
    except Exception as error:
        raise ValueError(
            f"Unable to load canonical base training config {base_config_path!r} "
            f"referenced by {config_path!r}: {error}"
        ) from error

    search_values = copy.deepcopy(config_values)
    search_values["base_config_path"] = base_config_path
    search_values.setdefault("project_root", base_config.project_root)
    try:
        search_config = CanonicalHyperparameterSearchConfig.model_validate(
            search_values
        )
        base_values = base_config.model_dump(mode="python")
        base_values["project_root"] = search_config.project_root
        search_config._compiled_config = _compile_value(
            search_config.overrides,
            base_values,
            (),
        )
        search_config.validate_compiled_search()
    except (ValidationError, ValueError, TypeError) as error:
        raise ValueError(
            f"Invalid canonical hyperparameter search config {config_path!r}:\n{error}"
        ) from error
    return search_config


__all__ = [
    "CanonicalHyperparameterSearchConfig",
    "compile_canonical_hyperparameter_search_config",
    "resolve_base_config_path",
]
