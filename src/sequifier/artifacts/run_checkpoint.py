"""Schema and atomic storage for exact-resume run checkpoints."""

from __future__ import annotations

import contextlib
import os
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from sequifier.artifacts.model_artifact import ModelArtifact

RUN_CHECKPOINT_FORMAT_VERSION = 2


@dataclass(frozen=True)
class OptimizationState:
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    scaler: dict[str, Any]
    optimizer_step: int
    skip_next_scheduler_step: bool


@dataclass(frozen=True)
class DistributedRandomState:
    states: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LoaderState:
    parts: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class IntegrationState:
    values: dict[str, Any]


@dataclass(frozen=True)
class RunCheckpoint:
    format_version: int
    model: ModelArtifact
    optimization: OptimizationState
    run_state: dict[str, Any]
    random_state: DistributedRandomState
    loader_state: LoaderState
    integration_state: IntegrationState
    training_config: Any

    def validate(self) -> None:
        if self.format_version != RUN_CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported run checkpoint format {self.format_version}; "
                f"expected {RUN_CHECKPOINT_FORMAT_VERSION}."
            )
        self.model.validate()

    def state_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "artifact_type": "sequifier_run_checkpoint",
            "format_version": self.format_version,
            "model": self.model.state_dict(),
            "optimization": asdict(self.optimization),
            "run_state": self.run_state,
            "random_state": {"states": self.random_state.states},
            "loader_state": {"parts": self.loader_state.parts},
            "integration_state": {"values": self.integration_state.values},
            "training_config": self.training_config.model_dump(mode="python"),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "RunCheckpoint":
        if payload.get("artifact_type") != "sequifier_run_checkpoint":
            raise ValueError("Unsupported checkpoint: expected new Sequifier format.")
        from sequifier.config.train_config import ResolvedSequifierConfig

        checkpoint = cls(
            format_version=int(payload["format_version"]),
            model=ModelArtifact.from_state_dict(payload["model"]),
            optimization=OptimizationState(**dict(payload["optimization"])),
            run_state=dict(payload["run_state"]),
            random_state=DistributedRandomState(
                states=tuple(payload["random_state"]["states"])
            ),
            loader_state=LoaderState(parts=dict(payload["loader_state"]["parts"])),
            integration_state=IntegrationState(
                values=dict(payload["integration_state"]["values"])
            ),
            training_config=ResolvedSequifierConfig.model_validate(
                payload["training_config"]
            ),
        )
        checkpoint.validate()
        return checkpoint


class RunCheckpointStore:
    """Own filesystem layout and atomic persistence of run checkpoints."""

    def __init__(self, training_config: Any, model_name: str) -> None:
        self.latest_path = checkpoint_path(training_config)
        self.model_name = model_name

    def path_for(self, suffix: str | None) -> Path:
        return (
            self.latest_path
            if suffix in {None, "latest"}
            else self.latest_path.with_name(f"{self.model_name}-{suffix}.pt")
        )

    def save(self, checkpoint: RunCheckpoint, suffix: str | None) -> Path:
        output_path = self.path_for(suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_run_checkpoint(checkpoint, output_path, self.latest_path)
        return output_path

    def load(self, path: str | Path) -> RunCheckpoint:
        return load_run_checkpoint(path)


def load_run_checkpoint(path: str | Path) -> RunCheckpoint:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Run checkpoint payload must be a mapping.")
    return RunCheckpoint.from_state_dict(payload)


def write_run_checkpoint(
    checkpoint: RunCheckpoint, output_path: Path, latest_path: Path
) -> None:
    payload = checkpoint.state_dict()
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output_path)
    finally:
        with contextlib.suppress(OSError):
            os.remove(temporary)
    if output_path == latest_path:
        return
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_temporary = latest_path.with_name(
        f".{latest_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(payload, latest_temporary)
        os.replace(latest_temporary, latest_path)
    finally:
        with contextlib.suppress(OSError):
            os.remove(latest_temporary)


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


def select_run_checkpoint(training_config: Any) -> Path | None:
    resume = training_config.global_training.resume
    if resume is None or resume.policy == "never":
        return None
    path = checkpoint_path(training_config)
    if path.is_file():
        return path
    if resume.policy == "required":
        raise FileNotFoundError(f"Required run checkpoint does not exist: {path}")
    return None
