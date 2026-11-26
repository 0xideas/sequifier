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
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, PrivateAttr, ValidationError, model_validator

from sequifier.config.composition import (
    ComponentPath,
    changes_component_variant,
    component_discriminator,
    load_composed_yaml_config,
    merge_component_patch,
)
from sequifier.config.naming import ConfigModel, RunSeed
from sequifier.config.train_config import SequifierConfig, load_train_config_with_source
from sequifier.config.training_resolution import (
    initialization_model_config,
    resolve_execution_settings,
)
from sequifier.fingerprints import file_checksum
from sequifier.typechecking import beartype


def _validate_training_candidate(
    values: dict[str, Any], load_model_config=initialization_model_config
) -> SequifierConfig:
    candidate = SequifierConfig.model_validate(values)
    resolve_execution_settings(candidate.model_copy(deep=True), load_model_config)
    return candidate


_MISSING = object()
_DISTRIBUTION_KEYS = frozenset({"low", "high", "step", "log", "type"})
_PRIMITIVE_CATEGORICAL_TYPES = (str, int, float, bool, type(None))


@beartype
def _path_name(path: ComponentPath) -> str:
    """Return a stable, readable Optuna parameter name."""

    value = ""
    for component in path:
        if isinstance(component, int):
            value += f"[{component}]"
        else:
            value += ("." if value else "") + component
    return value or "config"


@beartype
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
    path: ComponentPath
    kind: Literal["categorical", "int", "float"]
    choices: tuple[Any, ...] = ()
    low: int | float | None = None
    high: int | float | None = None
    step: int | float | None = None
    log: bool = False

    @property
    @beartype
    def name(self) -> str:
        return _path_name(self.path)

    @beartype
    def baseline(self) -> Any:
        if self.kind == "categorical":
            return copy.deepcopy(self.choices[0])
        return self.low

    @beartype
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

    @beartype
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

    @beartype
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

    @beartype
    def __init__(self, params: dict[str, Any]):
        self.params = params

    @beartype
    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        return self.params.get(name, choices[0])

    @beartype
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

    @beartype
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
    @beartype
    def materialize(self, trial: Any | None) -> Any:
        raise NotImplementedError

    def materialize_against(self, current: Any, trial: Any | None) -> Any:
        return self.materialize(trial)

    @beartype
    def spaces(self) -> list[_SearchSpace]:
        return []


@dataclass(frozen=True)
class _LiteralValue(_CompiledValue):
    value: Any

    @beartype
    def materialize(self, trial: Any | None) -> Any:
        return copy.deepcopy(self.value)


@dataclass(frozen=True)
class _SampledValue(_CompiledValue):
    space: _SearchSpace

    @beartype
    def materialize(self, trial: Any | None) -> Any:
        return self.space.baseline() if trial is None else self.space.sample(trial)

    @beartype
    def spaces(self) -> list[_SearchSpace]:
        return [self.space]


@dataclass(frozen=True)
class _MappingValue(_CompiledValue):
    path: ComponentPath
    base: dict[str, Any]
    children: dict[str, _CompiledValue]
    variants: _SampledValue | None = None

    @beartype
    def materialize(self, trial: Any | None) -> dict[str, Any]:
        return self.materialize_against(self.base, trial)

    @beartype
    def materialize_against(
        self,
        current: Any,
        trial: Any | None,
    ) -> dict[str, Any]:
        base = current if isinstance(current, dict) else {}
        result = (
            merge_component_patch(base, self.variants.materialize(trial), self.path)
            if self.variants
            else copy.deepcopy(base)
        )
        discriminator = component_discriminator(self.path)
        selectors: dict[str, Any] = dict(self.children)
        if discriminator is not None and discriminator in self.children:
            selectors[discriminator] = self.children[discriminator].materialize(trial)
        if changes_component_variant(result, selectors, self.path):
            result = {}
        for key, child in self.children.items():
            result[key] = (
                selectors[key]
                if key == discriminator
                else child.materialize_against(result.get(key, _MISSING), trial)
            )
        return result

    @beartype
    def spaces(self) -> list[_SearchSpace]:
        spaces = self.variants.spaces() if self.variants else []
        return spaces + [
            space for child in self.children.values() for space in child.spaces()
        ]


@dataclass(frozen=True)
class _ListValue(_CompiledValue):
    path: ComponentPath
    base: list[Any]
    children: dict[int, _CompiledValue]

    @beartype
    def materialize(self, trial: Any | None) -> list[Any]:
        return self.materialize_against(self.base, trial)

    @beartype
    def materialize_against(self, current: Any, trial: Any | None) -> list[Any]:
        if not isinstance(current, list):
            raise ValueError(
                f"{_path_name(self.path)} indexed parameters require a list in "
                "the selected variant"
            )
        result = copy.deepcopy(current)
        for index, child in self.children.items():
            if index >= len(result):
                raise ValueError(
                    f"{_path_name(self.path)} index {index} is outside the "
                    f"selected variant list of length {len(result)}"
                )
            result[index] = child.materialize_against(result[index], trial)
        return result

    @beartype
    def spaces(self) -> list[_SearchSpace]:
        return [space for child in self.children.values() for space in child.spaces()]


@beartype
def _categorical_space(path: ComponentPath, choices: Any) -> _SampledValue:
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"{_path_name(path)} choices must be a non-empty list")
    identities = [_candidate_identity(value) for value in choices]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{_path_name(path)} choices cannot contain duplicates")
    return _SampledValue(
        _SearchSpace(path=path, kind="categorical", choices=tuple(choices))
    )


@beartype
def _distribution_space(
    path: ComponentPath,
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


@beartype
def _is_distribution(expression: dict[Any, Any]) -> bool:
    keys = set(expression)
    return {"low", "high"} <= keys and keys <= _DISTRIBUTION_KEYS


@beartype
def _list_index_mapping(expression: dict[Any, Any]) -> dict[int, Any] | None:
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


@beartype
def _compile_value(expression: Any, base: Any, path: ComponentPath) -> _CompiledValue:
    if isinstance(expression, dict):
        if set(expression) == {"$choices"}:
            return _categorical_space(path, expression["$choices"])
        if set(expression) == {"$fixed"}:
            if not isinstance(expression["$fixed"], (dict, list)):
                raise ValueError(f"{_path_name(path)}: use primitive scalars directly")
            return _LiteralValue(expression["$fixed"])
        if set(expression) == {"$range"}:
            spec = expression["$range"]
            if not isinstance(spec, dict) or not _is_distribution(spec):
                raise ValueError(
                    f"{_path_name(path)}.$range requires low, high and optional step/log/type"
                )
            return _distribution_space(path, spec, base)
        if _is_distribution(expression) and not isinstance(base, dict):
            raise ValueError(
                f"{_path_name(path)}: wrap numeric distributions in $range"
            )
        unsupported = {
            key for key in expression if isinstance(key, str) and key.startswith("$")
        } - {"$variants"}
        if unsupported:
            raise ValueError(
                f"{_path_name(path)}: invalid search operator combination {sorted(unsupported)}"
            )

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
            return _ListValue(path, copy.deepcopy(base), children)

    if isinstance(expression, dict):
        base_mapping = copy.deepcopy(base) if isinstance(base, dict) else {}
        if isinstance(base, dict) and changes_component_variant(base, expression, path):
            base_mapping = {}
        variant_key = "$variants" if "$variants" in expression else None
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
            return _MappingValue(
                path,
                base_mapping,
                mapping_children,
                _categorical_space((*path, "__variant"), variants),
            )
        return _MappingValue(path, base_mapping, mapping_children)

    if isinstance(expression, list):
        raise ValueError(
            f"{_path_name(path)}: use $choices to sample or $fixed to replace a list"
        )

    return _LiteralValue(expression)


class SearchOutputSettings(ConfigModel):
    trial_config_directory: str = Field(min_length=1)


class SearchRunSettings(ConfigModel):
    project_root: str = Field(min_length=1)
    study_name: str = Field(min_length=1)
    outputs: SearchOutputSettings


def _monitor_identity(candidate, phase, monitor):
    identity = {
        "algorithm": phase.algorithm,
        "monitor": monitor.model_dump(mode="json") if monitor else None,
    }
    if hasattr(phase.evaluation, "environment"):
        environment = phase.evaluation.environment
        if isinstance(environment, str):
            environment = candidate.environments[environment]
        identity["environment"] = environment.model_dump(mode="json")
        identity["episodes"] = phase.evaluation.model_dump(
            mode="json",
            include={
                "seeds",
                "max_episode_steps",
                "timeout_seconds",
                "action_selection",
            },
        )
        if phase.algorithm == "grpo" and monitor.metric == "score_mean":
            identity["reward_scale"] = phase.reward_scale
    elif monitor:
        if phase.algorithm == "supervised":
            references = [monitor]
        else:
            references = phase.evaluation.sources or phase.sources
            identity["implicit_validation"] = phase.evaluation.sources is None
        sources = {}
        for ref in references:
            dataset = candidate.datasets[ref.dataset]
            view = dataset.views.get(ref.part)
            if dataset.partition:
                seed = candidate.run.reproducibility.seed
                if seed == "random":
                    raise ValueError(
                        "A generated evaluation partition requires a fixed run seed for search"
                    )
                identity["partition_seed"] = seed
            # Partition membership depends on the entire population before view filtering.
            parts = (
                view.parts
                if view and view.parts and not dataset.partition
                else dataset.sources
            )
            sources[ref.source] = {
                "view": view.model_dump(mode="json") if view else None,
                "partition": dataset.partition.model_dump(mode="json")
                if dataset.partition
                else None,
                "parts": {
                    key: dataset.sources[key].model_dump(mode="json") for key in parts
                },
            }
        identity["sources"] = sources
    return identity


class CanonicalHyperparameterSearchConfig(ConfigModel):
    """Search controls and a recursive sampler over one canonical train config."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        populate_by_name=True,
    )

    base_config_path: str = Field(min_length=1)
    parameters: dict[str, Any]
    run: SearchRunSettings
    method: Literal["bayesian", "random", "grid"] = "bayesian"
    sampler_seed: RunSeed | None = None
    trials: int | None = Field(default=None, gt=0)
    prune_trials: bool = True
    pruning_warmup_epochs: int | None = Field(default=None, ge=0)
    pruning_warmup_batches: int | None = Field(default=None, ge=0)
    pruning_warmup_iterations: int | None = Field(default=None, ge=0)

    evaluation_inference_config_path: str | None = None
    evaluation_script_path: str | None = None
    evaluation_metric_directions: list[Literal["min", "max"]] | None = None
    evaluation_metrics: list[str] | None = None
    _compiled_config: _CompiledValue = PrivateAttr()
    _objective_identity: dict | None = PrivateAttr(default=None)

    @property
    def objective_identity(self):
        if self.evaluation_script_path:
            root = Path(self.run.project_root)
            script = (root / self.evaluation_script_path).resolve()
            identity = {
                "script": str(script),
                "script_sha256": file_checksum(script),
                "metrics": self.evaluation_metrics,
            }
            if self.evaluation_inference_config_path:
                path = Path(self.evaluation_inference_config_path)
                path = (path if path.exists() else root / path).resolve()
                identity["inference_config"] = str(path)
                identity["inference_config_sha256"] = file_checksum(path)
            return identity
        return self._objective_identity

    @model_validator(mode="after")
    @beartype
    def validate_search_controls(self):
        reserved_parameters = {"model_name", "project_root"} & set(
            self.parameters.get("run", {})
        )
        if reserved_parameters:
            raise ValueError(
                "Search parameters cannot set run-controlled "
                f"fields: {sorted(reserved_parameters)}"
            )
        warmups = (
            self.pruning_warmup_epochs,
            self.pruning_warmup_batches,
            self.pruning_warmup_iterations,
        )
        if sum(value is not None for value in warmups) > 1:
            raise ValueError(
                "Select exactly one pruning warmup unit when configuring warmup"
            )
        if self.evaluation_metrics is None:
            if (
                self.evaluation_metric_directions is not None
                and len(self.evaluation_metric_directions) != 1
            ):
                raise ValueError(
                    "The final phase monitor requires one optimization direction"
                )
            if self.evaluation_script_path is not None:
                raise ValueError("evaluation_script_path requires evaluation_metrics")
        if self.evaluation_metrics is not None:
            if not self.evaluation_metrics:
                raise ValueError("evaluation_metrics cannot be empty")
            if self.evaluation_script_path is None:
                raise ValueError(
                    "evaluation_script_path must be provided if evaluation_metrics "
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
            if self.evaluation_inference_config_path is None:
                warnings.warn(
                    "Please provide evaluation_inference_config_path if your "
                    "evaluation_script_path requires inference outputs",
                    stacklevel=2,
                )
        if self.evaluation_script_path is not None and not os.path.exists(
            os.path.join(self.run.project_root, self.evaluation_script_path)
        ):
            raise ValueError(
                f"evaluation_script_path {self.evaluation_script_path!r} does not exist "
                f"under run.project_root {self.run.project_root!r}"
            )
        if self.evaluation_inference_config_path is not None:
            inference_path = self.evaluation_inference_config_path
            if not os.path.isabs(inference_path):
                inference_path = os.path.join(self.run.project_root, inference_path)
            if not os.path.exists(inference_path):
                raise ValueError(
                    f"evaluation_inference_config_path {self.evaluation_inference_config_path!r} "
                    "does not exist"
                )
        return self

    @property
    @beartype
    def search_spaces(self) -> tuple[_SearchSpace, ...]:
        return tuple(self._compiled_config.spaces())

    @beartype
    def grid_size(self) -> int:
        return math.prod(space.grid_size() for space in self.search_spaces)

    def _validate_trial_objective(self, candidate):
        phase = candidate.ordered_phases[-1]
        if (
            self.prune_trials
            and self.pruning_warmup_iterations
            and phase.algorithm == "supervised"
        ):
            raise ValueError(
                "pruning_warmup_iterations requires a policy objective phase"
            )
        if (
            self.prune_trials
            and self.pruning_warmup_batches
            and phase.algorithm != "supervised"
        ):
            raise ValueError("Use pruning_warmup_iterations for policy updates")
        if (
            self.prune_trials
            and self.pruning_warmup_epochs
            and phase.algorithm
            not in {
                "supervised",
                "policy_imitation",
            }
        ):
            raise ValueError(
                "Epoch warmup requires a supervised or imitation objective phase"
            )
        if self.evaluation_script_path is None:
            monitor = phase.evaluation.monitor if phase.evaluation else None
            direction = monitor.direction if monitor else "min"
            if self.evaluation_metric_directions is None:
                self.evaluation_metric_directions = [direction]
            elif self.evaluation_metric_directions != [direction]:
                raise ValueError(
                    "search direction must match the final phase evaluation monitor"
                )

            identity = _monitor_identity(candidate, phase, monitor)
            if self._objective_identity is None:
                self._objective_identity = identity
            elif self._objective_identity != identity:
                raise ValueError(
                    "Search candidates must use the same objective metric and evaluation source"
                )

    @beartype
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

        load_model_config = lru_cache(maxsize=1)(initialization_model_config)
        for description, trial in candidates:
            try:
                candidate = _validate_training_candidate(
                    self._materialized_values(trial, 0), load_model_config
                )
                self._validate_trial_objective(candidate)
            except (ValidationError, ValueError) as error:
                raise ValueError(
                    "Search parameters produce an invalid "
                    f"training config for {description}:\n{error}"
                ) from error

        if self.method == "grid":
            combinations = self.grid_size()
            if self.trials is not None and self.trials != combinations:
                raise ValueError(
                    "For method='grid', trials must equal the number "
                    f"of configured combinations ({combinations}), got "
                    f"{self.trials}. Remove trials to run the complete grid."
                )

    @beartype
    def _materialized_values(self, trial: Any | None, run_index: int) -> dict[str, Any]:
        values = self._compiled_config.materialize(trial)
        if not isinstance(values, dict):
            raise ValueError("Search parameters must produce a mapping")
        model_override = self.parameters.get("model")
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
                and "position_encoding_scope" not in architecture_override
            ):
                architecture = values["model"]["backbone"]["architecture"]
                if architecture["position_encoding"]["type"] in {
                    "range",
                    "range_concat",
                }:
                    architecture["position_encoding_scope"] = "global"
        values["run"]["project_root"] = self.run.project_root
        trial_count = self.trials if self.trials is not None else self.grid_size()
        values["run"]["model_name"] = (
            f"{self.run.study_name}-run-{run_index:0{len(str(trial_count))}d}"
        )
        outputs = values["run"].get("outputs", {})
        run_name = values["run"]["model_name"]

        def scope_file(settings, key):
            if settings.get(key):
                path = Path(settings[key])
                settings[key] = str(
                    path.with_name(f"{path.stem}-{run_name}{path.suffix}")
                )

        for key in ("model_path", "report_path"):
            scope_file(outputs, key)
        if outputs.get("phase_directory"):
            outputs["phase_directory"] = os.path.join(
                outputs["phase_directory"], run_name
            )
        recovery = values["run"].get("recovery", {}).get("resume") or {}
        scope_file(recovery, "checkpoint_path")
        for phase in values["phases"].values():
            collection = phase.get("collection") or {}
            scope_file(collection.get("output") or {}, "manifest_path")
        return values

    @beartype
    def sample_trial(self, trial: Any, run_index: int) -> SequifierConfig:
        """Sample and validate one concrete canonical authored training config."""

        values = self._materialized_values(trial, run_index)
        try:
            candidate = _validate_training_candidate(values)
            self._validate_trial_objective(candidate)
            return candidate
        except (ValidationError, ValueError) as error:
            try:
                parameters = trial.params
            except AttributeError:
                parameters = {}
            raise ValueError(
                "Sampled hyperparameters produce an invalid canonical training "
                f"config for parameters {parameters!r}:\n{error}"
            ) from error


@beartype
def resolve_base_config_path(config_path: str, base_config_path: str) -> str:
    """Resolve a base config relative to the hyperparameter-search entry file."""

    if os.path.isabs(base_config_path):
        return os.path.abspath(base_config_path)
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(config_path)), base_config_path)
    )


@beartype
def compile_canonical_hyperparameter_search_config(
    config_path: str,
    config_values: dict[str, Any],
    skip_metadata: bool,
) -> CanonicalHyperparameterSearchConfig:
    """Compile base-training plus recursive parameters into a canonical sampler."""

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
                {"project_root": config_values.get("run", {}).get("project_root")},
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
    if isinstance(search_values.get("run"), dict):
        search_values["run"].setdefault("project_root", base_config.run.project_root)
    try:
        base_values = load_composed_yaml_config(base_config_path)
        search_config = CanonicalHyperparameterSearchConfig.model_validate(
            search_values
        )
        base_values["run"]["project_root"] = search_config.run.project_root
        search_config._compiled_config = _compile_value(
            search_config.parameters,
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
