"""Public loading helpers for current Sequifier artifact formats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sequifier.artifacts.model_artifact import (
    load_model_artifact,
    load_weights_from_run_checkpoint,
)
from sequifier.model.parameter_catalog import ParameterCatalog


@dataclass(frozen=True)
class ExecutionOptions:
    device: str = "cpu"
    compile: bool = False
    training_mode: bool = False
    enable_grad: bool = True
    enable_dropout: bool = False


@dataclass
class LoadedModel:
    network: nn.Module
    config: Any
    parameter_catalog: ParameterCatalog
    artifact_metadata: dict[str, Any]


def load_model_for_analysis(
    path: str | Path,
    *,
    options: ExecutionOptions = ExecutionOptions(),
    config: Any = None,
    interface_name: str | None = None,
) -> LoadedModel:
    """Load a current artifact; external configuration is intentionally rejected."""

    if config is not None:
        raise ValueError("Current Sequifier artifacts embed their model configuration.")
    artifact_path = Path(path).expanduser().resolve()
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    artifact_type = payload.get("artifact_type") if isinstance(payload, dict) else None
    if artifact_type == "sequifier_model":
        network, resolved_config, artifact = load_model_artifact(
            artifact_path,
            device=options.device,
            interface_name=interface_name,
        )
    elif artifact_type == "sequifier_run_checkpoint":
        network, resolved_config, artifact = load_weights_from_run_checkpoint(
            artifact_path,
            device=options.device,
            interface_name=interface_name,
        )
    else:
        raise ValueError(f"Unsupported Sequifier artifact: {artifact_path}.")
    network.train(options.training_mode)
    for module in network.modules():
        if isinstance(module, nn.Dropout):
            module.train(options.enable_dropout)
    if not options.enable_grad:
        network.requires_grad_(False)
    catalog = ParameterCatalog(network)
    if options.compile:
        network = torch.compile(network)
    return LoadedModel(
        network=network,
        config=resolved_config,
        parameter_catalog=catalog,
        artifact_metadata={
            "path": str(artifact_path),
            "artifact_type": artifact_type,
            "format_version": artifact.format_version,
            "metadata": artifact.metadata,
            "selected_interface": interface_name,
        },
    )
