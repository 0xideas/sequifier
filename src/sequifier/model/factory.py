from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from sequifier.config.initialization_config import ModelInitializationConfig
from sequifier.helpers import get_torch_dtype
from sequifier.model.backbone import TransformerBackbone
from sequifier.model.decoders import build_target_decoding
from sequifier.model.execution import ModelExecution
from sequifier.model.ingestion_compiler import compile_feature_ingestion
from sequifier.model.ingestions import BaseFeatureIngestion, IngestionOutputProjection
from sequifier.model.initialization import (
    initialize_model_weights,
    initialize_owned_ingestion,
)
from sequifier.model.initialization_seed import derived_seed, isolated_initialization
from sequifier.model.layers import RMSNorm, SequifierEncoderLayer
from sequifier.model.network import ComposableTransformerNetwork, ModelInterfaceModule
from sequifier.model.static_conditioning import StaticConditionEncoder
from sequifier.objectives import BERTObjective, CausalObjective
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
    runtime_metadata: ModelRuntimeMetadata


def validate_hash_contracts(network: ComposableTransformerNetwork, config: Any) -> None:
    """Check persisted codec buffers after weights are restored."""
    from sequifier.model.ingestions import EmbeddingFeatureIngestion, MultiHashEmbedding

    for name, interface in ModelExecution.from_config(config).interfaces.items():
        route = network.resolve_interface(name)
        for column, expected in interface.categorical_hash_contracts.items():
            from sequifier.model.hash_codec import CategoricalHashCodec

            fresh = CategoricalHashCodec(
                interface.categorical_hashing[column],
                interface.n_classes[column],
                interface.target_decoder_ids.get(column),
            )
            if fresh.contract() != expected:
                raise ValueError(
                    f"Resolved hash contract for {column!r} differs from its configuration"
                )
            if column in interface.target_decoder_ids:
                codec = route.require_decoder().hash_codec(column)
                if codec.contract() != expected:
                    raise ValueError(
                        f"Persisted hash codec for {column!r} differs from its resolved contract"
                    )
            if column in interface.input_columns and expected["type"] == "multi_hash":
                for module in route.ingestion.modules():
                    if (
                        isinstance(module, EmbeddingFeatureIngestion)
                        and column in module.encoder
                    ):
                        embedding = module.encoder[column]
                        if isinstance(embedding, MultiHashEmbedding) and (
                            embedding.multipliers.tolist() != expected["multipliers"]
                            or embedding.offsets.tolist() != expected["offsets"]
                        ):
                            raise ValueError(
                                f"Persisted input hash coefficients for {column!r} differ from its resolved contract"
                            )


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
    layer_config = config.options.layer_type_dtypes
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
        elif (
            isinstance(
                module,
                (
                    nn.LayerNorm,
                    RMSNorm,
                    nn.GroupNorm,
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                ),
            )
            and "norm" in layer_config
        ):
            module.to(dtype=get_torch_dtype(layer_config["norm"]))


@dataclass(frozen=True)
class _ComponentInitialization:
    module: nn.Module
    policy: ModelInitializationConfig
    seed: int | None = None


@beartype
def _initialize_components(
    components: dict[str, _ComponentInitialization], logger: Any
) -> None:
    configured_targets: set[tuple[str, str]] = set()
    matched_targets: set[tuple[str, str]] = set()
    for component in components.values():
        module, initialization = component.module, component.policy
        configured_targets.update(initialization.configured_targets())
        with isolated_initialization(component.seed):
            if (
                isinstance(module, BaseFeatureIngestion)
                and module.initialization_config is not None
            ):
                matched = initialize_owned_ingestion(
                    module, initialization, warn_unmatched=False
                )
            else:
                matched = initialize_model_weights(
                    module, initialization, warn_unmatched=False
                )
        matched_targets.update(matched)
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
    persisted_decoder_ids = view.target_decoder_ids
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
    config: ModelExecution,
    *,
    device: torch.device,
    initialize: bool,
    restored_history_interface: str | None,
    logger: Any,
    encoder_interface: str | None = None,
) -> BuiltModel:
    from sequifier.config.train_config import interface_build_view

    interfaces = (
        config.interfaces
        if encoder_interface is None
        else {encoder_interface: config.interface(encoder_interface)}
    )
    backbone_seed = config.backbone.initialization_seed
    with isolated_initialization(derived_seed(backbone_seed, (), "constructor")):
        backbone = TransformerBackbone(config.backbone.architecture)
    backbone_initialization = _ComponentInitialization(
        backbone,
        config.backbone.initialization,
        derived_seed(backbone_seed, (), "initializer"),
    )

    routes = {}
    runtime_metadata = {}
    route_initialization_components: dict[str, _ComponentInitialization] = {}
    for name, interface in interfaces.items():
        view = interface_build_view(config, interface)
        load_pretrained = (
            initialize and device.type != "meta" and name != restored_history_interface
        )
        built_ingestion = compile_feature_ingestion(
            hparams=view,
            direct_real_dtype_provider=lambda: cast(
                SequifierEncoderLayer, backbone.layers[0]
            ).ff.get_first_layer_dtype(),
            max_concat_tensors=config.options.max_concat_tensors,
            load_pretrained=load_pretrained,
        )
        from sequifier.model.ingestions import EmbeddingFeatureIngestion

        applied_hash_columns = {
            column
            for module in built_ingestion.module.modules()
            if isinstance(module, EmbeddingFeatureIngestion)
            for column in module.hashing
        }
        missing_hash_inputs = (
            set(interface.categorical_hashing) & set(interface.input_columns)
        ) - applied_hash_columns
        if missing_hash_inputs:
            raise ValueError(
                "categorical_hashing input columns require an embedding ingestion "
                f"branch: {sorted(missing_hash_inputs)}"
            )
        with isolated_initialization(
            derived_seed(
                interface.ingestion.initialization_seed, ("adapter",), "constructor"
            )
        ):
            adapter: nn.Module = (
                nn.Identity()
                if built_ingestion.width == backbone.input_dim
                else IngestionOutputProjection(
                    built_ingestion.width, backbone.input_dim
                )
            )
        decoder = None
        if encoder_interface is None:
            metadata = _decoder_metadata(view)
            runtime_metadata[name] = metadata
            decoder = build_target_decoding(
                view,
                target_n_classes=metadata.target_n_classes,
                target_global_to_decoder=metadata.target_global_to_decoder,
            )
            for column, expected in interface.categorical_hash_contracts.items():
                actual = (
                    decoder.hash_codecs[column].contract()
                    if column in decoder.hash_codecs
                    else None
                )
                if column in interface.target_decoder_ids and actual != expected:
                    raise ValueError(
                        f"Persisted categorical hash contract for {column!r} differs "
                        "from the resolved decoder codec."
                    )
        static_encoder = None
        if interface.static_conditioning is not None:
            static_seed = interface.ingestion.initialization_seed
            with isolated_initialization(
                derived_seed(static_seed, ("static_conditioning",), "constructor")
            ):
                static_encoder = StaticConditionEncoder(
                    interface.static_conditioning.columns,
                    interface.static_categorical_columns,
                    interface.static_real_columns,
                    interface.n_classes,
                    backbone.model_dim,
                    interface.static_conditioning.total_embedding_dim,
                    interface.static_conditioning.feature_embedding_dims,
                )
        image_prefix_encoder = None
        if interface.preconditioning_image_encoder is not None:
            from sequifier.model.images import ImagePrefixEncoder

            image_source = interface.preconditioning_image
            if image_source is None:
                raise ValueError(
                    "Metadata lacks the configured preconditioning image source"
                )
            image_prefix_encoder = ImagePrefixEncoder(
                interface.preconditioning_image_encoder,
                backbone.model_dim,
                len(image_source.views),
                load_pretrained,
            )
        prefix_projection = (
            nn.Linear(2 * backbone.model_dim, backbone.model_dim)
            if static_encoder is not None and image_prefix_encoder is not None
            else None
        )
        routes[name] = ModelInterfaceModule(
            ingestion=built_ingestion.module,
            ingestion_adapter=adapter,
            decoder=decoder,
            decoder_input_width=interface.decoder.support_length * backbone.model_dim,
            decoding_support=interface.decoder.support_length,
            prediction_length=interface.decoder.prediction_length,
            target_columns=tuple(interface.target_columns),
            target_column_types=dict(interface.target_column_types),
            static_condition_encoder=static_encoder,
            preconditioning_image_encoder=image_prefix_encoder,
            prefix_projection=prefix_projection,
            depth_layouts=interface.depth_layouts,
            input_columns=tuple(interface.input_columns),
            input_n_classes=interface.n_classes,
        )
        from sequifier.model.prediction_distribution import PredictionScales

        distributions = interface.prediction_distributions
        if encoder_interface is None and distributions:
            routes[name].prediction_scales = PredictionScales(distributions).to(device)
        route_initialization_components.update(
            {
                f"interfaces.{name}.ingestion": _ComponentInitialization(
                    built_ingestion.module,
                    interface.ingestion.initialization,
                    built_ingestion.module.initialization_seed,
                ),
                f"interfaces.{name}.ingestion_adapter": _ComponentInitialization(
                    adapter,
                    interface.ingestion.initialization,
                    derived_seed(
                        interface.ingestion.initialization_seed,
                        ("adapter",),
                        "initializer",
                    ),
                ),
            }
        )
        if decoder is not None:
            route_initialization_components[f"interfaces.{name}.decoder"] = (
                _ComponentInitialization(
                    decoder,
                    interface.decoder.initialization,
                )
            )
        if static_encoder is not None:
            route_initialization_components[
                f"interfaces.{name}.static_condition_encoder"
            ] = _ComponentInitialization(
                static_encoder,
                interface.ingestion.initialization,
                derived_seed(
                    interface.ingestion.initialization_seed,
                    ("static_conditioning",),
                    "initializer",
                ),
            )
        if image_prefix_encoder is not None:
            route_initialization_components[
                f"interfaces.{name}.preconditioning_image_encoder"
            ] = _ComponentInitialization(
                image_prefix_encoder, interface.ingestion.initialization
            )
        if prefix_projection is not None:
            route_initialization_components[f"interfaces.{name}.prefix_projection"] = (
                _ComponentInitialization(
                    prefix_projection,
                    interface.ingestion.initialization,
                )
            )

    attention_objective = (
        CausalObjective if config.options.attention == "causal" else BERTObjective
    )
    network = ComposableTransformerNetwork(
        backbone=backbone,
        interfaces=routes,
        attention_mask_policy=attention_objective.build_attention_mask_policy(
            config.options.context_length
        ),
        attention_mask_policy_with_prefix=attention_objective.build_attention_mask_policy(
            config.options.context_length + 1
        ),
        context_length=config.options.context_length,
    )
    if initialize:
        if len(config.interfaces) == 1:
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
                "backbone": backbone_initialization,
            }
            for component in (
                "decoder",
                "static_condition_encoder",
                "preconditioning_image_encoder",
                "prefix_projection",
            ):
                key = f"interfaces.{name}.{component}"
                if key in route_initialization_components:
                    initialization_components[key] = route_initialization_components[
                        key
                    ]
        else:
            initialization_components = {
                "backbone": backbone_initialization,
                **route_initialization_components,
            }
        _initialize_components(initialization_components, logger)
    _apply_layer_dtypes(network, config)
    network.to(device)
    return BuiltModel(
        network=network,
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
    restored_history_interface: str | None = None,
    logger: Any | None = None,
) -> BuiltModel:
    """Build one shared backbone and every distinct named model interface."""

    resolved_logger: Any = logger if logger is not None else logging.getLogger(__name__)
    return _build_composable_network(
        ModelExecution.from_config(config),
        device=device,
        initialize=initialize,
        restored_history_interface=restored_history_interface,
        logger=resolved_logger,
    )


def build_history_encoder(
    config: Any,
    interface_name: str,
    *,
    device: torch.device,
    restored_history: bool,
) -> ComposableTransformerNetwork:
    """Build the selected ingestion path and backbone, using the actor's namespaces."""
    return _build_composable_network(
        ModelExecution.from_config(config),
        device=device,
        initialize=True,
        restored_history_interface=interface_name if restored_history else None,
        logger=logging.getLogger(__name__),
        encoder_interface=interface_name,
    ).network
