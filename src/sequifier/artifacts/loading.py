from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sequifier.artifacts.model_config import resolved_config_from_model_config
from sequifier.config.composable_train_config import (
    ResolvedSequifierConfig as TrainModel,
)
from sequifier.config.composable_train_config import load_train_config
from sequifier.model.factory import build_transformer_network
from sequifier.model.parameter_catalog import ParameterCatalog
from sequifier.typechecking import beartype


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


@beartype
def _resolve_config(
    payload: dict[str, Any],
    config: TrainModel | str | Path | None,
    *,
    device: str,
    interface_name: str | None,
) -> tuple[Any, str | None]:
    model_config = payload.get("model_config")
    if model_config is not None:
        return resolved_config_from_model_config(
            model_config,
            device=device,
            interface_name=interface_name,
        )
    embedded = payload.get("training_config")
    if embedded is not None:
        return TrainModel.model_validate(embedded), interface_name
    if isinstance(config, TrainModel):
        return config.model_copy(deep=True), interface_name
    if config is not None:
        return load_train_config(str(config), {}, skip_metadata=False), interface_name
    raise ValueError(
        "The artifact has no embedded resolved training_config; provide config= "
        "when loading a legacy run checkpoint."
    )


@beartype
def load_model_for_analysis(
    path: str | Path,
    *,
    options: ExecutionOptions = ExecutionOptions(),
    config: TrainModel | str | Path | None = None,
    interface_name: str | None = None,
) -> LoadedModel:
    artifact_path = Path(path).expanduser().resolve()
    payload = torch.load(
        artifact_path,
        map_location=torch.device(options.device),
        weights_only=False,
    )
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Unsupported Sequifier artifact: {artifact_path}.")

    resolved_config, selected_interface = _resolve_config(
        payload,
        config,
        device=options.device,
        interface_name=interface_name,
    )
    resolved_config.device = options.device
    resolved_config.training_spec.torch_compile = "none"
    built = build_transformer_network(
        resolved_config,
        device=torch.device(options.device),
        initialize=False,
    )
    network: nn.Module = built.network
    state = {
        name.replace("_orig_mod.", ""): value
        for name, value in payload["model_state_dict"].items()
    }
    network.load_state_dict(state)

    if options.training_mode:
        network.train()
    else:
        network.eval()
    for module in network.modules():
        if isinstance(module, nn.Dropout):
            module.train(options.enable_dropout)
    if not options.enable_grad:
        network.requires_grad_(False)

    catalog = ParameterCatalog(network)
    if options.compile:
        network = torch.compile(network)

    metadata = {
        key: value
        for key, value in payload.items()
        if key
        not in {"model_state_dict", "optimizer_state_dict", "best_model_state_dict"}
    }
    metadata["path"] = str(artifact_path)
    metadata["selected_interface"] = selected_interface
    return LoadedModel(
        network=network,
        config=resolved_config,
        parameter_catalog=catalog,
        artifact_metadata=metadata,
    )
