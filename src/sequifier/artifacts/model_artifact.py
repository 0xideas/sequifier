"""Portable, execution-only Sequifier model artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from sequifier.artifacts.model_config import (
    model_execution_from_model_config,
    validate_interface_versions,
)
from sequifier.artifacts.model_export import model_execution_config
from sequifier.artifacts.state_dict import (
    canonicalize_state_dict,
    validate_model_state_contract,
)
from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.fingerprints import fingerprint
from sequifier.io.atomic import atomic_save
from sequifier.model.execution import ModelExecution
from sequifier.model.factory import build_transformer_network


@dataclass(frozen=True)
class ModelExecutionConfig:
    """Serializable configuration required to reconstruct model execution."""

    values: dict[str, Any]

    @classmethod
    def from_training_config(cls, config: Any) -> "ModelExecutionConfig":
        return cls(values=model_execution_config(config))

    @property
    def identity(self) -> str:
        """Fingerprint the portable contract with its execution order and typed labels."""

        def component_values(component):
            values = dict(component)
            if branches := values.get("branches"):
                values["branches"] = [
                    (name, component_values(branch))
                    for name, branch in branches.items()
                ]
            for name in ("groups", "image_cells"):
                if isinstance(values.get(name), dict):
                    values[name] = list(values[name].items())
            return values

        interfaces = []
        for name, interface in self.values["interfaces"].items():
            values = dict(interface)
            for component in ("ingestion", "decoder"):
                values[component] = component_values(interface[component])
            if layouts := interface.get("feature_layout"):
                values["feature_layout"] = {
                    key: {
                        **layout,
                        "axes": list(layout["axes"].items()),
                        "columns": list(layout["columns"].items()),
                    }
                    for key, layout in layouts.items()
                }
            if policy := interface.get("policy"):
                values["policy"] = {
                    **policy,
                    "actions": list(policy["actions"].items()),
                }
            # JSON object keys would conflate an integer label with its string spelling.
            values["id_maps"] = {
                column: sorted(
                    labels.items(), key=lambda item: (type(item[0]).__name__, item[0])
                )
                for column, labels in interface["id_maps"].items()
            }
            interfaces.append((name, values))
        return fingerprint({**self.values, "interfaces": interfaces})


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
        require_contract_version(
            "portable model artifact",
            self.format_version,
            ContractVersions.MODEL_ARTIFACT,
        )
        for interface in self.model_config.values.get("interfaces", {}).values():
            validate_interface_versions(interface)
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
        require_contract_version(
            "portable model artifact",
            payload.get("format_version"),
            ContractVersions.MODEL_ARTIFACT,
        )
        artifact = cls(
            format_version=payload["format_version"],
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
    from sequifier.config.train_config import ResolvedSequifierConfig

    if isinstance(config, ResolvedSequifierConfig) and config.training_plan:
        resolved_provenance["training_plan"] = [
            phase.model_dump(mode="json") for phase in config.training_plan
        ]
    artifact = ModelArtifact(
        format_version=ContractVersions.MODEL_ARTIFACT,
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
) -> tuple[Any, ModelExecution, ModelArtifact]:
    """Load a portable artifact and return its network and execution contract."""

    artifact, _ = read_artifact(path, expected_type="sequifier_model")
    return restore_model_artifact(
        artifact, device=device, interface_name=interface_name
    )


def restore_model_artifact(
    artifact: ModelArtifact,
    *,
    device: str = "cpu",
    interface_name: str | None = None,
) -> tuple[Any, ModelExecution, ModelArtifact]:
    """Reconstruct execution identically for portable models and run checkpoints."""
    config, _ = model_execution_from_model_config(
        artifact.model_config.values, interface_name=interface_name
    )
    return restore_execution_network(artifact, config, device), config, artifact


def restore_execution_network(
    artifact: ModelArtifact, config: ModelExecution, device: str
):
    """Populate a model from its validated execution contract and portable weights."""
    network = build_transformer_network(
        config, device=torch.device(device), initialize=False
    )
    network.load_state_dict(artifact.model_state_dict)
    from sequifier.model.factory import validate_hash_contracts

    validate_hash_contracts(network, config)
    return network


def read_artifact(
    path: str | Path, *, expected_type: str | None = None
) -> tuple[ModelArtifact, str]:
    """Read and validate an execution artifact, retaining its source kind."""
    payload = torch.load(
        Path(path).expanduser().resolve(), map_location="cpu", weights_only=False
    )
    artifact_type = payload.get("artifact_type") if isinstance(payload, dict) else None
    if artifact_type in {
        "sequifier_run_checkpoint",
        "sequifier_policy_run",
        "sequifier_learning_run",
    }:
        from sequifier.artifacts.checkpoint_envelope import validate_checkpoint_envelope

        validate_checkpoint_envelope(payload)
        payload = payload.get("model")
        if payload is None:
            raise ValueError("Checkpoint precedes the first executable model")
    if expected_type is not None and artifact_type != expected_type:
        raise ValueError(f"Expected {expected_type} artifact, received {artifact_type}")
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
) -> tuple[Any, ModelExecution, ModelArtifact]:
    return restore_model_artifact(
        model_artifact_from_run_checkpoint(path),
        device=device,
        interface_name=interface_name,
    )
