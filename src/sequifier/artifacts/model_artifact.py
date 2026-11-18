"""Portable, execution-only Sequifier model artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from sequifier.artifacts.model_config import (
    model_execution_from_model_config,
    resolved_training_config,
)
from sequifier.artifacts.model_export import model_execution_config
from sequifier.artifacts.state_dict import (
    canonicalize_state_dict,
    validate_model_state_contract,
)
from sequifier.io.atomic import atomic_save
from sequifier.model.execution import ModelExecution
from sequifier.model.factory import build_transformer_network

MODEL_ARTIFACT_FORMAT_VERSION = 3


@dataclass(frozen=True)
class ModelExecutionConfig:
    """Serializable configuration required to reconstruct model execution."""

    values: dict[str, Any]

    @classmethod
    def from_training_config(cls, config: Any) -> "ModelExecutionConfig":
        return cls(values=model_execution_config(config))


@dataclass(frozen=True)
class ModelArtifactMetadata:
    trace_sites: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelArtifact:
    format_version: int
    model_config: ModelExecutionConfig
    model_state_dict: dict[str, Tensor]
    metadata: ModelArtifactMetadata

    def validate(self) -> None:
        if self.format_version != MODEL_ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported model artifact format {self.format_version}; "
                f"expected {MODEL_ARTIFACT_FORMAT_VERSION}."
            )
        for name, interface in self.model_config.values.get("interfaces", {}).items():
            if interface.get("depth_layouts") and "execution_schema" not in interface:
                raise ValueError(
                    f"Depth model interface {name!r} requires an execution schema"
                )
        validate_model_state_contract(self.model_state_dict)

    def state_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "artifact_type": "sequifier_model",
            "format_version": self.format_version,
            "model_config": self.model_config.values,
            "model_state_dict": self.model_state_dict,
            "metadata": asdict(self.metadata),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "ModelArtifact":
        if payload.get("artifact_type") != "sequifier_model":
            raise ValueError("Artifact is not a portable Sequifier model.")
        artifact = cls(
            format_version=int(payload["format_version"]),
            model_config=ModelExecutionConfig(dict(payload["model_config"])),
            model_state_dict=canonicalize_state_dict(payload["model_state_dict"]),
            metadata=ModelArtifactMetadata(**dict(payload.get("metadata", {}))),
        )
        artifact.validate()
        return artifact


def build_model_artifact(
    network: Any,
    config: Any,
    *,
    state_dict: Mapping[str, Tensor] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> ModelArtifact:
    state = canonicalize_state_dict(
        network.state_dict() if state_dict is None else state_dict
    )
    trace_sites: list[str] = []
    for interface_name in network.interfaces:
        trace_sites.extend(
            f"{interface_name}:{site.name}"
            for site in network.trace_catalog_for(interface_name)
        )
    resolved_provenance = dict(provenance or {})
    phases = getattr(config, "training_plan", None)
    if phases:
        resolved_provenance["training_plan"] = [
            phase.model_dump(mode="json") for phase in phases
        ]
    artifact = ModelArtifact(
        format_version=MODEL_ARTIFACT_FORMAT_VERSION,
        model_config=ModelExecutionConfig.from_training_config(config),
        model_state_dict=state,
        metadata=ModelArtifactMetadata(
            trace_sites=tuple(trace_sites), provenance=resolved_provenance
        ),
    )
    artifact.validate()
    return artifact


def save_model_artifact(artifact: ModelArtifact, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    atomic_save(destination, artifact.state_dict())
    return destination


def load_model_artifact(
    path: str | Path,
    *,
    device: str = "cpu",
    interface_name: str | None = None,
) -> tuple[Any, Any, ModelArtifact]:
    """Load a portable artifact and return network, resolved config, and schema."""

    artifact, _ = read_artifact(path, expected_type="sequifier_model")
    network, execution, artifact = restore_model_artifact(
        artifact, device=device, interface_name=interface_name
    )
    return network, resolved_training_config(execution), artifact


def restore_model_artifact(
    artifact: ModelArtifact,
    *,
    device: str = "cpu",
    interface_name: str | None = None,
) -> tuple[Any, ModelExecution, ModelArtifact]:
    """Reconstruct execution identically for portable models and run checkpoints."""
    config, _ = model_execution_from_model_config(
        artifact.model_config.values,
        device=device,
        interface_name=interface_name,
    )
    built = build_transformer_network(
        config, device=torch.device(device), initialize=False
    )
    built.network.load_state_dict(artifact.model_state_dict)
    from sequifier.model.factory import validate_hash_contracts

    validate_hash_contracts(built.network, config)
    return built.network, config, artifact


def read_artifact(
    path: str | Path, *, expected_type: str | None = None
) -> tuple[ModelArtifact, str]:
    """Read and validate an execution artifact, retaining its source kind."""
    payload = torch.load(
        Path(path).expanduser().resolve(), map_location="cpu", weights_only=False
    )
    artifact_type = payload.get("artifact_type") if isinstance(payload, dict) else None
    if expected_type is not None and artifact_type != expected_type:
        raise ValueError(f"Expected {expected_type} artifact, received {artifact_type}")
    if artifact_type == "sequifier_run_checkpoint":
        from sequifier.artifacts.checkpoint_envelope import CHECKPOINT_FORMAT_VERSION

        if payload.get("format_version") not in {3, CHECKPOINT_FORMAT_VERSION}:
            raise ValueError("Unsupported run checkpoint envelope version")
        payload = payload.get("model")
        if payload is None:
            raise ValueError("Checkpoint precedes the first executable model")
    if not isinstance(payload, dict) or not isinstance(artifact_type, str):
        raise ValueError(
            "Sequifier model artifact must be a mapping with an artifact_type"
        )
    return ModelArtifact.from_state_dict(payload), artifact_type


def read_execution_artifact(path: str | Path) -> ModelArtifact:
    """Read the portable execution payload from either supported artifact kind."""
    return read_artifact(path)[0]


def load_execution_model(
    path: str | Path | ModelArtifact,
    *,
    device: str = "cpu",
    interface_name: str | None = None,
) -> tuple[Any, ModelExecution, ModelArtifact]:
    """Load a network and its model-owned execution contract."""
    return restore_model_artifact(
        path if isinstance(path, ModelArtifact) else read_execution_artifact(path),
        device=device,
        interface_name=interface_name,
    )


def model_artifact_from_run_checkpoint(path: str | Path) -> ModelArtifact:
    return read_artifact(path, expected_type="sequifier_run_checkpoint")[0]


def load_weights_from_run_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    interface_name: str | None = None,
) -> tuple[Any, Any, ModelArtifact]:
    network, execution, artifact = restore_model_artifact(
        model_artifact_from_run_checkpoint(path),
        device=device,
        interface_name=interface_name,
    )
    return network, resolved_training_config(execution), artifact
