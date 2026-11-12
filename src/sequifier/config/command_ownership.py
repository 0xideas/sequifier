"""Shared run settings and CLI overrides for command configurations."""

from copy import deepcopy

from pydantic import Field

from sequifier.config.ownership import (
    ExecutionSettings,
    ReproducibilitySettings,
    Settings,
)


class CommandExecutionSettings(ExecutionSettings):
    batch_size: int = Field(default=1, gt=0)


class CommandRunSettings(Settings):
    project_root: str = "."
    execution: CommandExecutionSettings = Field(
        default_factory=CommandExecutionSettings
    )
    reproducibility: ReproducibilitySettings = Field(
        default_factory=ReproducibilitySettings
    )

    def validate_for(self, command):
        execution = self.execution
        defaults = CommandExecutionSettings()
        for field in ("distribution", "precision", "compilation", "max_concat_tensors"):
            if getattr(execution, field) != getattr(defaults, field):
                raise ValueError(f"{command} does not support run.execution.{field}")
        loading = execution.data_loading
        if command == "infer" and loading != defaults.data_loading:
            raise ValueError("inference does not support data_loading controls")
        if command == "preprocess":
            if (
                execution.device != "cpu"
                or execution.batch_size != 1
                or self.reproducibility.enforce_determinism
            ):
                raise ValueError(
                    "preprocessing supports run execution workers and a seed; device/batching/determinism controls are unsupported"
                )
            other = loading.model_dump(exclude={"num_workers"})
            if other != defaults.data_loading.model_dump(exclude={"num_workers"}):
                raise ValueError("preprocessing only supports data_loading.num_workers")
        return self


def apply_run_overrides(values, overrides):
    """CLI scalar conveniences write the same canonical paths as YAML."""
    values, overrides = deepcopy(values), dict(overrides)
    run = values.setdefault("run", {})
    for key in ("project_root", "model_name"):
        if key in overrides and overrides[key] is not None:
            run[key] = overrides.pop(key)
    if "seed" in overrides and overrides["seed"] is not None:
        run.setdefault("reproducibility", {})["seed"] = overrides.pop("seed")
    if "device" in overrides and overrides["device"] is not None:
        run.setdefault("execution", {})["device"] = overrides.pop("device")
    if (
        "enforce_determinism" in overrides
        and overrides["enforce_determinism"] is not None
    ):
        run.setdefault("reproducibility", {})["enforce_determinism"] = overrides.pop(
            "enforce_determinism"
        )
    return values, {key: value for key, value in overrides.items() if value is not None}
