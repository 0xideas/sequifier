"""Schema and atomic storage for exact-resume run checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from sequifier.artifacts.model_artifact import ModelArtifact
from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.io.atomic import atomic_save


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
        require_contract_version(
            "supervised recovery",
            self.format_version,
            ContractVersions.SUPERVISED_RECOVERY,
        )
        self.model.validate()

    def state_dict(self) -> dict[str, Any]:
        self.validate()
        from sequifier.artifacts.checkpoint_envelope import checkpoint_envelope

        recovery = {
            "artifact_type": "sequifier_run_checkpoint",
            "recovery_type": "supervised",
            "format_version": self.format_version,
            "model": self.model.state_dict(),
            "optimization": asdict(self.optimization),
            "run_state": self.run_state,
            "random_state": {"states": self.random_state.states},
            "loader_state": {"parts": self.loader_state.parts},
            "integration_state": {"values": self.integration_state.values},
            "initialization_policy_version": ContractVersions.INITIALIZATION_POLICY,
            "initialization_seed_derivation_version": ContractVersions.INITIALIZATION_SEED_DERIVATION,
            "training_config": self.training_config.model_dump(mode="python"),
        }
        return checkpoint_envelope("supervised", recovery, self.model)

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "RunCheckpoint":
        from sequifier.artifacts.checkpoint_envelope import recovery_payload

        payload = recovery_payload(payload, "supervised")
        require_contract_version(
            "supervised recovery",
            payload.get("format_version"),
            ContractVersions.SUPERVISED_RECOVERY,
        )
        if payload.get("artifact_type") != "sequifier_run_checkpoint":
            raise ValueError("Expected a supervised checkpoint recovery payload")
        if payload.get("recovery_type") != "supervised":
            raise ValueError("Checkpoint belongs to another training runner")
        from sequifier.artifacts.model_config import validate_interface_storage_versions
        from sequifier.config.metadata import validate_persisted_metadata_versions
        from sequifier.config.train_config import ResolvedSequifierConfig

        require_contract_version(
            "initialization policy",
            payload.get("initialization_policy_version"),
            ContractVersions.INITIALIZATION_POLICY,
        )
        require_contract_version(
            "initialization seed derivation",
            payload.get("initialization_seed_derivation_version"),
            ContractVersions.INITIALIZATION_SEED_DERIVATION,
        )
        model = ModelArtifact.from_state_dict(payload["model"])
        saved_config = payload["training_config"]
        for interface in saved_config["interfaces"].values():
            validate_interface_storage_versions(interface)
        for dataset in saved_config["dataset_training"].values():
            if "interface" in dataset:
                validate_interface_storage_versions(dataset["interface"])
            for part in dataset["parts"].values():
                validate_persisted_metadata_versions(part["metadata"])
        training_config = ResolvedSequifierConfig.model_validate(saved_config)
        loader_parts = payload["loader_state"]["parts"]
        checkpoint = cls(
            format_version=payload["format_version"],
            model=model,
            optimization=OptimizationState(**dict(payload["optimization"])),
            run_state=dict(payload["run_state"]),
            random_state=DistributedRandomState(
                states=tuple(payload["random_state"]["states"])
            ),
            loader_state=LoaderState(parts=dict(loader_parts)),
            integration_state=IntegrationState(
                values=dict(payload["integration_state"]["values"])
            ),
            training_config=training_config,
        )
        checkpoint.validate()
        return checkpoint


class RunCheckpointStore:
    """Own filesystem layout and atomic persistence of run checkpoints."""

    def __init__(self, training_config: Any, model_name: str) -> None:
        self.latest_path = checkpoint_path(training_config)
        self.model_name = model_name
        self.phase_directory = training_config.phase_directory
        self.phases = training_config.training_plan

    def path_for(self, suffix: str | None, phase_index: int = 0) -> Path:
        if suffix in {None, "latest"}:
            return self.latest_path
        from sequifier.artifacts.phase import PhaseArtifacts

        directory = (
            PhaseArtifacts.for_phase(
                self.phase_directory, phase_index, self.phases[phase_index].name
            ).model.parent
            / "checkpoints"
        )
        return directory / f"{self.model_name}-{suffix}.pt"

    def save(self, checkpoint: RunCheckpoint, suffix: str | None) -> Path:
        output_path = self.path_for(suffix, checkpoint.run_state["phase_index"])
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
    atomic_save(output_path, payload)
    if output_path != latest_path:
        atomic_save(latest_path, payload)


def checkpoint_path(training_config: Any) -> Path:
    resume = training_config.runtime_settings.resume
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
    resume = training_config.runtime_settings.resume
    if resume is None or resume.policy == "never":
        return None
    path = checkpoint_path(training_config)
    if path.is_file():
        return path
    if resume.policy == "required":
        raise FileNotFoundError(f"Required run checkpoint does not exist: {path}")
    return None
