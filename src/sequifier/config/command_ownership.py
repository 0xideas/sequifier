"""Shared run settings and CLI overrides for command configurations."""

from copy import deepcopy

from pydantic import Field

from sequifier.config.naming import RunSeed
from sequifier.config.ownership import ReproducibilitySettings, Settings


class CommandRunSettings(Settings):
    project_root: str = "."


class InferenceExecutionSettings(Settings):
    device: str = "cpu"
    batch_size: int = Field(default=1, gt=0)


class InferenceRunSettings(CommandRunSettings):
    execution: InferenceExecutionSettings = Field(
        default_factory=InferenceExecutionSettings
    )
    reproducibility: ReproducibilitySettings = Field(
        default_factory=ReproducibilitySettings
    )


class PreprocessingSeed(Settings):
    seed: RunSeed = 1010


class SeededPreprocessingRun(CommandRunSettings):
    reproducibility: PreprocessingSeed = Field(default_factory=PreprocessingSeed)


class PreprocessingDataLoading(Settings):
    num_workers: int | None = Field(default=None, ge=0)


class PreprocessingExecution(Settings):
    data_loading: PreprocessingDataLoading = Field(
        default_factory=PreprocessingDataLoading
    )


class PreprocessingRunSettings(SeededPreprocessingRun):
    execution: PreprocessingExecution = Field(default_factory=PreprocessingExecution)


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
    # The CLI uses an explicit None to clear either column selection.
    return values, {
        key: value
        for key, value in overrides.items()
        if value is not None or key in {"selected_columns", "input_columns"}
    }
