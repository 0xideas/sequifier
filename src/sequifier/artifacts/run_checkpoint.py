import contextlib
import os
import uuid
from pathlib import Path
from typing import Any

import torch

from sequifier.typechecking import beartype


@beartype
def run_checkpoint_payload(
    *,
    checkpoint_metadata: dict[str, Any],
    epoch: int,
    batch: int,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any],
    scheduler_state_dict: dict[str, Any],
    scaler_state_dict: dict[str, Any],
    rng_state: Any,
    data_loader_generator_states: dict[str, Any],
    run_id: str,
    best_val_loss: float,
    n_epochs_no_improvement: int,
    best_model_state_dict: dict[str, Any] | None,
    backbone_parent_revision_id: str | None,
    loss: Any,
    training_config: dict[str, Any],
    training_state: dict[str, Any] | None,
    integration_state: dict[str, Any],
) -> dict[str, Any]:
    """Build the stable exact-resume checkpoint payload."""

    return {
        "checkpoint_metadata": checkpoint_metadata,
        "epoch": epoch,
        "batch": batch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "scaler_state_dict": scaler_state_dict,
        "rng_state": rng_state,
        "data_loader_generator_states": data_loader_generator_states,
        "run_id": run_id,
        "best_val_loss": best_val_loss,
        "n_epochs_no_improvement": n_epochs_no_improvement,
        "best_model_state_dict": best_model_state_dict,
        "backbone_parent_revision_id": backbone_parent_revision_id,
        "loss": loss,
        "training_config": training_config,
        "training_state": training_state,
        "integration_state": integration_state,
    }


@beartype
def write_run_checkpoint(
    payload: dict[str, Any], output_path: Path, latest_path: Path
) -> None:
    """Atomically write a run checkpoint and refresh its latest alias."""

    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output_path)
    finally:
        with contextlib.suppress(OSError):
            os.remove(temporary)
    if output_path == latest_path:
        return
    latest_temporary = latest_path.with_name(
        f".{latest_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(payload, latest_temporary)
        os.replace(latest_temporary, latest_path)
    finally:
        with contextlib.suppress(OSError):
            os.remove(latest_temporary)


@beartype
def checkpoint_path(training_config: Any) -> Path:
    resume = training_config.global_training.resume
    configured_path = resume.checkpoint_path if resume is not None else None
    if configured_path:
        path = Path(configured_path)
        if not path.is_absolute():
            path = Path(training_config.project_root) / path
    else:
        path = (
            Path(training_config.project_root)
            / "checkpoints"
            / "runs"
            / training_config.model_name
            / f"{training_config.model_name}-latest.pt"
        )
    return path.resolve()


@beartype
def select_run_checkpoint(training_config: Any) -> dict[str, Any] | None:
    resume = training_config.global_training.resume
    if resume is None or resume.policy == "never":
        return None
    path = checkpoint_path(training_config)
    if path.is_file():
        return {"path": str(path)}
    if resume.policy == "required":
        raise FileNotFoundError(f"Required run checkpoint does not exist: {path}")
    return None
