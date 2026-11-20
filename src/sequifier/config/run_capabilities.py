"""Runner and publication capabilities checked when compiling a training plan."""

import os
from typing import Literal

from sequifier.config.ownership import RunSettings

RunMode = Literal["supervised", "policy", "coordinated"]


def validate_runner_settings(
    run: RunSettings, *, supervised: bool, policy: bool
) -> None:
    execution = run.execution
    if policy:
        # These settings are consumed only by the supervised runner or finalizer.
        supervised_only = (
            (
                "run.execution",
                execution,
                {"distribution", "compilation", "precision", "max_concat_tensors"},
            ),
            (
                "run.execution.data_loading",
                execution.data_loading,
                {"num_workers", "load_full_data_to_ram", "max_ram_gb"},
            ),
            (
                "run.export",
                run.export,
                {
                    "onnx",
                    "embedding_model",
                    "with_dropout",
                    "generative_model",
                    "pt",
                    "batch_size",
                },
            ),
        )
        for path, settings, fields in supervised_only:
            if unsupported := fields.intersection(
                settings.model_dump(exclude_defaults=True)
            ):
                raise ValueError(
                    f"policy phases do not support {path} settings: {sorted(unsupported)}"
                )
        if not supervised and "log_interval_batches" in run.logging.model_dump(
            exclude_defaults=True
        ):
            raise ValueError("policy phases do not support batch logging intervals")
    elif (
        not (run.export.generative_model or run.export.embedding_model)
        and os.getenv("SEQUIFIER_PREVENT_EXPORT") is None
    ):
        raise ValueError("At least one model export must be enabled")


def training_run_mode(phases) -> RunMode:
    policy = [phase for phase in phases if phase.algorithm != "supervised"]
    if not policy:
        return "supervised"
    if len(policy) == len(phases) and len({p.model_interface for p in policy}) == 1:
        return "policy"
    return "coordinated"
