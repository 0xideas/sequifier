from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from sequifier.helpers import get_torch_dtype
from sequifier.model.backbone import TransformerBackbone
from sequifier.model.decoders import build_target_decoding
from sequifier.model.ingestion_compiler import compile_feature_ingestion
from sequifier.model.initialization import initialize_model_weights
from sequifier.model.layers import RMSNorm
from sequifier.model.network import ComposableTransformerNetwork, ModelInterfaceModule
from sequifier.objectives import CausalObjective, create_objective
from sequifier.special_tokens import resolve_categorical_decoder_ids
from sequifier.typechecking import beartype


@dataclass(frozen=True)
class InterfaceRuntimeMetadata:
    target_decoder_ids: dict[str, list[int]]
    target_n_classes: dict[str, int]
    target_global_to_decoder: dict[str, list[int]]


@dataclass(frozen=True)
class ModelRuntimeMetadata:
    interfaces: dict[str, InterfaceRuntimeMetadata]
    device: torch.device

    @beartype
    def _only(self) -> InterfaceRuntimeMetadata:
        if len(self.interfaces) != 1:
            raise ValueError("A model interface selection is required")
        return next(iter(self.interfaces.values()))

    @property
    @beartype
    def target_decoder_ids(self) -> dict[str, list[int]]:
        return self._only().target_decoder_ids

    @property
    @beartype
    def target_n_classes(self) -> dict[str, int]:
        return self._only().target_n_classes

    @property
    @beartype
    def target_global_to_decoder(self) -> dict[str, list[int]]:
        return self._only().target_global_to_decoder


@dataclass
class BuiltModel:
    network: ComposableTransformerNetwork
    objectives: dict[str, Any]
    runtime_metadata: ModelRuntimeMetadata


@beartype
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


@beartype
def _apply_layer_dtypes(network: nn.Module, config: Any) -> None:
    layer_config = config.global_training.layer_type_dtypes
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


@beartype
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


@beartype
def _decoder_metadata(view: Any) -> InterfaceRuntimeMetadata:
    categorical_targets = {
        column
        for column in view.target_columns
        if view.target_column_types[column] == "categorical"
    }
    persisted_decoder_ids = getattr(view, "target_decoder_ids", None)
    if persisted_decoder_ids:
        if set(persisted_decoder_ids) != categorical_targets:
            raise ValueError(
                "Persisted target_decoder_ids must contain exactly the categorical "
                f"targets; expected {sorted(categorical_targets)}, found "
                f"{sorted(persisted_decoder_ids)}."
            )
        target_decoder_ids = {
            column: [int(global_id) for global_id in persisted_decoder_ids[column]]
            for column in view.target_columns
            if column in categorical_targets
        }
        for column, global_ids in target_decoder_ids.items():
            if not global_ids:
                raise ValueError(
                    f"Persisted target_decoder_ids[{column!r}] cannot be empty."
                )
            if len(global_ids) != len(set(global_ids)):
                raise ValueError(
                    f"Persisted target_decoder_ids[{column!r}] contains duplicates."
                )
            invalid_ids = [
                global_id
                for global_id in global_ids
                if global_id < 0 or global_id >= view.n_classes[column]
            ]
            if invalid_ids:
                raise ValueError(
                    f"Persisted target_decoder_ids[{column!r}] contains IDs outside "
                    f"[0, {view.n_classes[column]}): {invalid_ids}."
                )
    else:
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


@beartype
def _build_composable_network(
    config: Any,
    *,
    device: torch.device,
    initialize: bool,
    logger: Any,
) -> BuiltModel:
    from sequifier.config.train_config import interface_build_view

    resolved_interfaces = {}
    for dataset in config.dataset_training.values():
        resolved_interfaces.setdefault(dataset.model_interface, dataset.interface)
    objectives = {}
    for name, interface in resolved_interfaces.items():
        objective_view = interface_build_view(config, interface)
        # A lean inference bundle may omit next-occurrence loss metadata. Its
        # forward attention policy remains the ordinary causal policy.
        if (
            config.global_training.training_objective == "next_occurrence"
            and config.global_training.next_occurrence_config is None
        ):
            objectives[name] = CausalObjective(objective_view)
        else:
            objectives[name] = create_objective(objective_view)
    objective = next(iter(objectives.values()))
    backbone = TransformerBackbone(config.model.backbone.architecture)

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
            device_max_concat_length=config.global_training.device_max_concat_length,
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
            config.global_training.context_length
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
                    config.model.backbone.initialization,
                ),
                f"interfaces.{name}.decoder": route_initialization_components[
                    f"interfaces.{name}.decoder"
                ],
            }
        else:
            initialization_components = {
                "backbone": (
                    backbone,
                    config.model.backbone.initialization,
                ),
                **route_initialization_components,
            }
        _initialize_components(initialization_components, logger)
    _apply_layer_dtypes(network, config)
    network.to(device)
    return BuiltModel(
        network=network,
        objectives=objectives,
        runtime_metadata=ModelRuntimeMetadata(
            interfaces=runtime_metadata,
            device=device,
        ),
    )


@beartype
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
