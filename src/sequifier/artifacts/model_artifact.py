"""Portable, execution-only Sequifier model artifacts."""

from __future__ import annotations

import contextlib
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from sequifier.artifacts.model_config import resolved_config_from_model_config
from sequifier.artifacts.model_export import model_execution_config
from sequifier.artifacts.state_dict import (
    canonicalize_state_dict,
    validate_model_state_contract,
)
from sequifier.model.factory import build_transformer_network

MODEL_ARTIFACT_FORMAT_VERSION = 1


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
    artifact = ModelArtifact(
        format_version=MODEL_ARTIFACT_FORMAT_VERSION,
        model_config=ModelExecutionConfig.from_training_config(config),
        model_state_dict=state,
        metadata=ModelArtifactMetadata(
            trace_sites=tuple(trace_sites), provenance=dict(provenance or {})
        ),
    )
    artifact.validate()
    return artifact


def save_model_artifact(artifact: ModelArtifact, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(artifact.state_dict(), temporary)
        os.replace(temporary, destination)
    finally:
        with contextlib.suppress(OSError):
            os.remove(temporary)
    return destination


def load_model_artifact(
    path: str | Path,
    *,
    device: str = "cpu",
    interface_name: str | None = None,
) -> tuple[Any, Any, ModelArtifact]:
    """Load a portable artifact and return network, resolved config, and schema."""

    payload = torch.load(
        Path(path).expanduser().resolve(),
        map_location=torch.device(device),
        weights_only=False,
    )
    artifact = ModelArtifact.from_state_dict(payload)
    config, _ = resolved_config_from_model_config(
        artifact.model_config.values,
        device=device,
        interface_name=interface_name,
    )
    built = build_transformer_network(
        config, device=torch.device(device), initialize=False
    )
    built.network.load_state_dict(artifact.model_state_dict)
    return built.network, config, artifact


def model_artifact_from_run_checkpoint(path: str | Path) -> ModelArtifact:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("artifact_type") != "sequifier_run_checkpoint":
        raise ValueError("Artifact is not a Sequifier run checkpoint.")
    return ModelArtifact.from_state_dict(payload["model"])


def load_weights_from_run_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    interface_name: str | None = None,
) -> tuple[Any, Any, ModelArtifact]:
    artifact = model_artifact_from_run_checkpoint(path)
    config, _ = resolved_config_from_model_config(
        artifact.model_config.values,
        device=device,
        interface_name=interface_name,
    )
    built = build_transformer_network(
        config, device=torch.device(device), initialize=False
    )
    built.network.load_state_dict(artifact.model_state_dict)
    return built.network, config, artifact
