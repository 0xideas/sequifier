"""Construction of pure Sequifier transformer networks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from sequifier.helpers import get_torch_dtype
from sequifier.model.backbone import TransformerBackbone
from sequifier.model.decoders import build_target_decoding
from sequifier.model.freezing import apply_model_freezing
from sequifier.model.ingestion_compiler import compile_feature_ingestion
from sequifier.model.initialization import initialize_model_weights
from sequifier.model.layers import RMSNorm
from sequifier.model.network import TransformerNetwork
from sequifier.objectives import create_objective
from sequifier.special_tokens import resolve_categorical_decoder_ids


@dataclass(frozen=True)
class ModelRuntimeMetadata:
    target_decoder_ids: dict[str, list[int]]
    target_n_classes: dict[str, int]
    target_global_to_decoder: dict[str, list[int]]
    device: torch.device


@dataclass
class BuiltModel:
    network: TransformerNetwork
    objective: Any
    runtime_metadata: ModelRuntimeMetadata


def _apply_layer_dtypes(network: nn.Module, config: Any) -> None:
    layer_config = config.training_spec.layer_type_dtypes
    if not layer_config:
        return
    for name, module in network.named_modules():
        if isinstance(module, nn.Linear):
            if name.startswith("decoder.") and "decoder" in layer_config:
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


def build_transformer_network(
    config: Any,
    *,
    device: torch.device,
    initialize: bool = True,
    apply_freezing: bool = True,
    logger: Any | None = None,
) -> BuiltModel:
    """Build model computation without optimizer or run-lifecycle state."""

    resolved_logger: Any = logger if logger is not None else logging.getLogger(__name__)
    objective = create_objective(config)
    architecture = config.model_spec.backbone.architecture
    backbone = TransformerBackbone(architecture)
    built_ingestion = compile_feature_ingestion(
        hparams=config,
        direct_real_dtype_provider=lambda: backbone.layers[
            0
        ].ff.get_first_layer_dtype(),
        device_max_concat_length=config.training_spec.device_max_concat_length,
    )
    ingestion_adapter: nn.Module = (
        nn.Identity()
        if built_ingestion.width == backbone.input_dim
        else nn.Linear(built_ingestion.width, backbone.input_dim)
    )
    ingestion_adapter._sequifier_layer_group = (  # type: ignore[attr-defined]
        "ingestion.output_projection"
    )

    target_decoder_ids = resolve_categorical_decoder_ids(
        config.target_columns,
        config.target_column_types,
        config.n_classes,
        getattr(config, "categorical_decoder_special_tokens", {}),
    )
    target_n_classes = {column: len(ids) for column, ids in target_decoder_ids.items()}
    target_global_to_decoder: dict[str, list[int]] = {}
    for column, ids in target_decoder_ids.items():
        inverse = {global_id: decoder_id for decoder_id, global_id in enumerate(ids)}
        target_global_to_decoder[column] = [
            inverse.get(global_id, -1) for global_id in range(config.n_classes[column])
        ]

    decoder = build_target_decoding(config, target_n_classes=target_n_classes)
    network = TransformerNetwork(
        ingestion=built_ingestion.module,
        ingestion_adapter=ingestion_adapter,
        backbone=backbone,
        decoder=decoder,
        attention_mask_policy=objective.build_attention_mask_policy(
            config.window_view.context_length
        ),
        decoding_support=config.model_spec.decoder.support,
        prediction_length=config.model_spec.decoder.prediction_length,
        target_columns=tuple(config.target_columns),
        target_column_types=dict(config.target_column_types),
    )

    if initialize:
        component_initializers = {
            "ingestion": config.model_spec.ingestion.initialization,
            "ingestion_adapter": config.model_spec.ingestion.initialization,
            "backbone": config.model_spec.backbone.initialization,
            "decoder": config.model_spec.decoder.initialization,
        }
        configured_targets: set[tuple[str, str]] = set()
        matched_targets: set[tuple[str, str]] = set()
        for component_name, initialization in component_initializers.items():
            configured_targets.update(initialization.configured_targets())
            matched_targets.update(
                initialize_model_weights(
                    getattr(network, component_name),
                    initialization,
                    warn_unmatched=False,
                )
            )
        unmatched = configured_targets.difference(matched_targets)
        if unmatched:
            targets = ", ".join(f"{group}.{kind}" for group, kind in sorted(unmatched))
            resolved_logger.warning(
                f"Initialization overrides matched no parameters: {targets}"
            )

    if apply_freezing:
        components = {
            "ingestion": (
                config.model_spec.ingestion,
                (network.ingestion, network.ingestion_adapter),
            ),
            "backbone": (config.model_spec.backbone, (network.backbone,)),
            "decoder": (config.model_spec.decoder, (network.decoder,)),
        }
        for component_name, (component_config, modules) in components.items():
            if not component_config.has_freezing_policy:
                continue
            matched_groups: set[str] = set()
            for module in modules:
                result = apply_model_freezing(
                    module,
                    freezing=component_config.freezing,
                    freezing_except=component_config.freezing_except,
                    warn_unmatched=False,
                )
                matched_groups.update(result.matched_groups)
            configured = set(
                component_config.freezing
                if component_config.freezing is not None
                else component_config.freezing_except or []
            )
            unmatched = configured.difference(matched_groups)
            if unmatched and component_config.freezing_except is not None:
                raise ValueError(
                    f"{component_name} freezing_except groups matched no parameters: "
                    f"{', '.join(sorted(unmatched))}"
                )
            if unmatched:
                resolved_logger.warning(
                    f"{component_name} freezing groups matched no parameters: "
                    f"{', '.join(sorted(unmatched))}"
                )
        if not any(parameter.requires_grad for parameter in network.parameters()):
            raise ValueError(
                "The configured freezing policies leave the model with no "
                "trainable parameters."
            )

    _apply_layer_dtypes(network, config)
    network.to(device)
    return BuiltModel(
        network=network,
        objective=objective,
        runtime_metadata=ModelRuntimeMetadata(
            target_decoder_ids=target_decoder_ids,
            target_n_classes=target_n_classes,
            target_global_to_decoder=target_global_to_decoder,
            device=device,
        ),
    )
