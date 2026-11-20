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


def read_training_plan(saved: dict):
    """Validate persisted storage versions before loading the resolved training plan."""
    from sequifier.artifacts.model_config import validate_interface_storage_versions
    from sequifier.config.metadata import validate_persisted_metadata_versions
    from sequifier.config.train_config import ResolvedSequifierConfig

    execution = saved["execution"]
    interfaces = execution["interfaces"]
    for name, storage in execution["interface_storage"].items():
        validate_interface_storage_versions(storage, interfaces[name]["depth_layouts"])
    for dataset in saved["dataset_training"].values():
        if (storage := dataset.get("storage")) is not None:
            interface = interfaces[dataset["model_interface"]]
            validate_interface_storage_versions(storage, interface["depth_layouts"])
        for part in dataset["parts"].values():
            validate_persisted_metadata_versions(part["metadata"])
    return ResolvedSequifierConfig.model_validate(saved)


def omit_scheduler_budget_assertions(phases):
    """OneCycleLR total_steps is checked against the data-derived budget at preflight."""
    for phase in phases:
        for role in phase["optimization"].values():
            scheduler = role["scheduler"]
            if scheduler["name"] == "OneCycleLR":
                scheduler["arguments"].pop("total_steps", None)


def training_plan_identity(config):
    """Fingerprint the resolved plan independently of how recovery was requested."""
    from sequifier.fingerprints import fingerprint

    values = config.model_dump(mode="json")
    values["runtime_settings"].pop("resume", None)
    values.get("ownership", {}).get("run", {}).get("recovery", {}).pop("resume", None)
    omit_scheduler_budget_assertions(values["training_plan"])
    omit_scheduler_budget_assertions(
        values.get("ownership", {}).get("training_plan", {}).get("phases", ())
    )
    return fingerprint(values)


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
    dependencies: dict[str, str]

    def validate(self) -> None:
        require_contract_version(
            "supervised recovery",
            self.format_version,
            ContractVersions.SUPERVISED_RECOVERY,
        )
        self.model.validate()

    def phase_boundary(self):
        from sequifier.artifacts.checkpoint_envelope import (
            phase_boundary,
            selected_model_identity,
        )

        state = self.run_state
        completed = state["phase_commits"]
        active = (
            {
                "index": state.get("phase_offset", 0) + state["phase_index"],
                "transaction": state["optimizer_step"],
            }
            if len(completed) <= state["phase_index"]
            else None
        )
        return phase_boundary(
            state["run_id"], selected=selected_model_identity(completed), active=active
        )

    def state_dict(self) -> dict[str, Any]:
        from sequifier.artifacts.checkpoint_envelope import checkpoint_envelope

        self.validate()
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
            "dependencies": self.dependencies,
            "training_config": self.training_config.model_dump(mode="python"),
        }
        return checkpoint_envelope(
            "supervised", recovery, self.model, boundary=self.phase_boundary()
        )

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "RunCheckpoint":
        from sequifier.artifacts.checkpoint_envelope import (
            recovery_payload,
            require_phase_boundary,
        )

        envelope = payload
        payload = recovery_payload(envelope, "supervised")
        require_contract_version(
            "supervised recovery",
            payload.get("format_version"),
            ContractVersions.SUPERVISED_RECOVERY,
        )
        if payload.get("artifact_type") != "sequifier_run_checkpoint":
            raise ValueError("Expected a supervised checkpoint recovery payload")
        if payload.get("recovery_type") != "supervised":
            raise ValueError("Checkpoint belongs to another training runner")

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
        training_config = read_training_plan(payload["training_config"])
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
            dependencies=dict(payload["dependencies"]),
        )
        checkpoint.validate()
        require_phase_boundary(envelope, checkpoint.phase_boundary())
        return checkpoint


class RunCheckpointStore:
    """Own filesystem layout and atomic persistence of run checkpoints."""

    def __init__(self, training_config: Any, model_name: str) -> None:
        self.latest_path = checkpoint_path(training_config)
        self.model_name = model_name
        self.phase_directory = training_config.phase_directory
        self.phases = training_config.training_plan

    def path_for(
        self, suffix: str | None, phase_index: int = 0, *, phase_offset: int = 0
    ) -> Path:
        if suffix in {None, "latest"}:
            return self.latest_path
        from sequifier.artifacts.phase import PhaseArtifacts

        directory = (
            PhaseArtifacts.for_phase(
                self.phase_directory,
                phase_offset + phase_index,
                self.phases[phase_index].name,
            ).model.parent
            / "checkpoints"
        )
        return directory / f"{self.model_name}-{suffix}.pt"

    def save(self, checkpoint: RunCheckpoint, suffix: str | None) -> Path:
        output_path = self.path_for(
            suffix,
            checkpoint.run_state["phase_index"],
            phase_offset=checkpoint.run_state.get("phase_offset", 0),
        )
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


def checkpoint_path(training_config: Any, *, resume=None) -> Path:
    if resume is None:
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


def select_run_checkpoint(training_config: Any, *, resume=None) -> Path | None:
    if resume is None:
        resume = training_config.runtime_settings.resume
    if resume is None or resume.policy == "never":
        return None
    path = checkpoint_path(training_config, resume=resume)
    if path.is_file():
        return path
    if resume.policy == "required":
        raise FileNotFoundError(f"Required run checkpoint does not exist: {path}")
    return None


def read_resume_seed(run) -> int | None:
    """Read the saved run seed before resolving seed-dependent inputs."""
    from sequifier.artifacts.checkpoint_envelope import validate_checkpoint_envelope

    path = select_run_checkpoint(run, resume=run.recovery.resume)
    if path is None:
        return None
    envelope = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_envelope(envelope)
    recovery = envelope["recovery"]
    if envelope["recovery_type"] == "supervised":
        require_contract_version(
            "supervised recovery",
            recovery.get("format_version"),
            ContractVersions.SUPERVISED_RECOVERY,
        )
        recovery = recovery["training_config"]["execution"]
    elif envelope["recovery_type"] == "policy":
        recovery = recovery["plan"]
    seed = recovery.get("seed")
    if not isinstance(seed, int):
        raise ValueError(
            f"Checkpoint {path} has no valid run seed. "
            "Set run.reproducibility.seed to the original integer."
        )
    return seed
