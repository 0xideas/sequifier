from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from sequifier.helpers import get_torch_dtype
from sequifier.model.backbone import TransformerBackbone
from sequifier.model.decoders import build_target_decoding
from sequifier.model.ingestion_compiler import compile_feature_ingestion
from sequifier.model.initialization import initialize_model_weights
from sequifier.model.layers import RMSNorm
from sequifier.model.network import ComposableTransformerNetwork, ModelInterfaceModule
from sequifier.objectives import CausalObjective, create_objective
from sequifier.special_tokens import resolve_categorical_decoder_ids


@dataclass(frozen=True)
class InterfaceRuntimeMetadata:
    target_decoder_ids: dict[str, list[int]]
    target_n_classes: dict[str, int]
    target_global_to_decoder: dict[str, list[int]]


@dataclass(frozen=True)
class ModelRuntimeMetadata:
    interfaces: dict[str, InterfaceRuntimeMetadata]
    device: torch.device

    def _only(self) -> InterfaceRuntimeMetadata:
        if len(self.interfaces) != 1:
            raise ValueError("A model interface selection is required")
        return next(iter(self.interfaces.values()))

    @property
    def target_decoder_ids(self) -> dict[str, list[int]]:
        return self._only().target_decoder_ids

    @property
    def target_n_classes(self) -> dict[str, int]:
        return self._only().target_n_classes

    @property
    def target_global_to_decoder(self) -> dict[str, list[int]]:
        return self._only().target_global_to_decoder


@dataclass
class BuiltModel:
    network: ComposableTransformerNetwork
    objective: Any
    runtime_metadata: ModelRuntimeMetadata


def compile_unique_layers(layers: nn.ModuleList) -> None:
    """Compile each distinct layer once while preserving shared-layer aliases."""

    compiled_layers: dict[int, nn.Module] = {}
    for index, layer in enumerate(layers):
        layer_id = id(layer)
        compiled_layer = compiled_layers.get(layer_id)
        if compiled_layer is None:
            compiled_layer = torch.compile(layer)
            compiled_layers[layer_id] = compiled_layer
        layers[index] = compiled_layer


def compile_composable_training_model(model: Any, config: Any) -> nn.Module:
    """Compile a canonical training model according to configured datasets."""

    if config.global_training_spec.torch_compile == "none":
        return model
    if len(config.dataset_training_spec) == 1:
        if config.global_training_spec.torch_compile == "inner":
            compile_unique_layers(model.layers)
            return model
        return torch.compile(model)

    model.backbone = torch.compile(model.backbone)
    model.network.backbone = model.backbone
    for route in model.interfaces.values():
        route.ingestion = torch.compile(route.ingestion)
        if next(route.ingestion_adapter.parameters(), None) is not None:
            route.ingestion_adapter = torch.compile(route.ingestion_adapter)
        route.decoder = torch.compile(route.decoder)
    return model


def wrap_composable_ddp(model: Any, config: Any, local_rank: int) -> nn.Module | None:
    """Apply whole-model or component DDP wrapping for canonical training."""

    device_ids = [local_rank] if config.device.startswith("cuda") else None
    if len(config.dataset_training_spec) == 1:
        return DDP(model, device_ids=device_ids, find_unused_parameters=False)

    model.backbone = DDP(
        model.backbone, device_ids=device_ids, find_unused_parameters=False
    )
    model.network.backbone = model.backbone
    for route in model.interfaces.values():
        route.ingestion = DDP(
            route.ingestion, device_ids=device_ids, find_unused_parameters=False
        )
        if next(route.ingestion_adapter.parameters(), None) is not None:
            route.ingestion_adapter = DDP(
                route.ingestion_adapter,
                device_ids=device_ids,
                find_unused_parameters=False,
            )
        route.decoder = DDP(
            route.decoder, device_ids=device_ids, find_unused_parameters=False
        )
    return None


def _training_spec(config: Any) -> Any:
    return config.global_training_spec


def _apply_layer_dtypes(network: nn.Module, config: Any) -> None:
    layer_config = _training_spec(config).layer_type_dtypes
    if not layer_config:
        return
    for name, module in network.named_modules():
        if isinstance(module, nn.Linear):
            is_decoder = name.startswith("decoder.") or ".decoder." in name
            if is_decoder and "decoder" in layer_config:
                module.to(dtype=get_torch_dtype(layer_config["decoder"]))
            elif "linear" in layer_config:
                module.to(dtype=get_torch_dtype(layer_config["linear"]))
        elif isinstance(module, nn.Embedding) and "embedding" in layer_config:
            module.to(dtype=get_torch_dtype(layer_config["embedding"]))
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            dtype_name = (
                layer_config.get("conv")
                or layer_config.get("linear")
                or layer_config.get("embedding")
            )
            if dtype_name is not None:
                module.to(dtype=get_torch_dtype(dtype_name))
        elif isinstance(module, nn.MultiheadAttention) and "linear" in layer_config:
            module.to(dtype=get_torch_dtype(layer_config["linear"]))
        elif isinstance(module, (nn.LayerNorm, RMSNorm)) and "norm" in layer_config:
            module.to(dtype=get_torch_dtype(layer_config["norm"]))


def _initialize_components(
    components: dict[str, tuple[nn.Module, Any]], logger: Any
) -> None:
    configured_targets: set[tuple[str, str]] = set()
    matched_targets: set[tuple[str, str]] = set()
    for module, initialization in components.values():
        configured_targets.update(initialization.configured_targets())
        matched_targets.update(
            initialize_model_weights(
                module,
                initialization,
                warn_unmatched=False,
            )
        )
    unmatched = configured_targets.difference(matched_targets)
    if unmatched:
        targets = ", ".join(f"{group}.{kind}" for group, kind in sorted(unmatched))
        logger.warning(f"Initialization overrides matched no parameters: {targets}")


def _decoder_metadata(view: Any) -> InterfaceRuntimeMetadata:
    target_decoder_ids = resolve_categorical_decoder_ids(
        view.target_columns,
        view.target_column_types,
        view.n_classes,
        view.categorical_decoder_special_tokens,
    )
    target_n_classes = {column: len(ids) for column, ids in target_decoder_ids.items()}
    target_global_to_decoder = {}
    for column, ids in target_decoder_ids.items():
        inverse = {global_id: decoder_id for decoder_id, global_id in enumerate(ids)}
        target_global_to_decoder[column] = [
            inverse.get(global_id, -1) for global_id in range(view.n_classes[column])
        ]
    return InterfaceRuntimeMetadata(
        target_decoder_ids=target_decoder_ids,
        target_n_classes=target_n_classes,
        target_global_to_decoder=target_global_to_decoder,
    )


def _build_composable_network(
    config: Any,
    *,
    device: torch.device,
    initialize: bool,
    logger: Any,
) -> BuiltModel:
    from sequifier.config.composable_train_config import interface_build_view

    resolved_interfaces = {}
    for dataset in config.dataset_training_spec.values():
        resolved_interfaces.setdefault(dataset.model_interface, dataset.interface)
    first_interface = next(iter(resolved_interfaces.values()))
    objective_view = interface_build_view(config, first_interface)
    # A lean inference bundle intentionally omits next-occurrence loss
    # configuration and categorical ID maps.  Its forward attention policy is
    # the ordinary causal policy; training configs still construct the complete
    # objective below.
    if (
        config.global_training_spec.training_objective == "next_occurrence"
        and config.global_training_spec.next_occurrence_config is None
    ):
        objective = CausalObjective(objective_view)
    else:
        objective = create_objective(objective_view)
    backbone = TransformerBackbone(config.model_spec.backbone.architecture)

    routes = {}
    runtime_metadata = {}
    route_initialization_components: dict[str, tuple[nn.Module, Any]] = {}
    for name, interface in resolved_interfaces.items():
        view = interface_build_view(config, interface)
        built_ingestion = compile_feature_ingestion(
            hparams=view,
            direct_real_dtype_provider=lambda: backbone.layers[
                0
            ].ff.get_first_layer_dtype(),
            device_max_concat_length=config.global_training_spec.device_max_concat_length,
        )
        adapter: nn.Module = (
            nn.Identity()
            if built_ingestion.width == backbone.input_dim
            else nn.Linear(built_ingestion.width, backbone.input_dim)
        )
        adapter._sequifier_layer_group = "ingestion.output_projection"  # type: ignore[attr-defined]
        metadata = _decoder_metadata(view)
        decoder = build_target_decoding(
            view, target_n_classes=metadata.target_n_classes
        )
        routes[name] = ModelInterfaceModule(
            ingestion=built_ingestion.module,
            ingestion_adapter=adapter,
            decoder=decoder,
            decoding_support=interface.decoder.support,
            prediction_length=interface.decoder.prediction_length,
            target_columns=tuple(interface.target_columns),
            target_column_types=dict(interface.target_column_types),
        )
        runtime_metadata[name] = metadata
        route_initialization_components.update(
            {
                f"interfaces.{name}.ingestion": (
                    built_ingestion.module,
                    interface.ingestion.initialization,
                ),
                f"interfaces.{name}.ingestion_adapter": (
                    adapter,
                    interface.ingestion.initialization,
                ),
                f"interfaces.{name}.decoder": (
                    decoder,
                    interface.decoder.initialization,
                ),
            }
        )

    network = ComposableTransformerNetwork(
        backbone=backbone,
        interfaces=routes,
        attention_mask_policy=objective.build_attention_mask_policy(
            config.global_training_spec.context_length
        ),
    )
    if initialize:
        if len(routes) == 1:
            # Preserve the legacy single-interface seeded initialization order.
            # The order is observable because default initializers consume RNG.
            name = next(iter(routes))
            initialization_components = {
                f"interfaces.{name}.ingestion": route_initialization_components[
                    f"interfaces.{name}.ingestion"
                ],
                f"interfaces.{name}.ingestion_adapter": (
                    route_initialization_components[
                        f"interfaces.{name}.ingestion_adapter"
                    ]
                ),
                "backbone": (
                    backbone,
                    config.model_spec.backbone.initialization,
                ),
                f"interfaces.{name}.decoder": route_initialization_components[
                    f"interfaces.{name}.decoder"
                ],
            }
        else:
            initialization_components = {
                "backbone": (
                    backbone,
                    config.model_spec.backbone.initialization,
                ),
                **route_initialization_components,
            }
        _initialize_components(initialization_components, logger)
    _apply_layer_dtypes(network, config)
    network.to(device)
    return BuiltModel(
        network=network,
        objective=objective,
        runtime_metadata=ModelRuntimeMetadata(
            interfaces=runtime_metadata,
            device=device,
        ),
    )


def build_transformer_network(
    config: Any,
    *,
    device: torch.device,
    initialize: bool = True,
    logger: Any | None = None,
) -> BuiltModel:
    """Build one shared backbone and every distinct named model interface."""

    resolved_logger: Any = logger if logger is not None else logging.getLogger(__name__)
    return _build_composable_network(
        config,
        device=device,
        initialize=initialize,
        logger=resolved_logger,
    )
