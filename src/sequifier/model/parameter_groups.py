"""Ordered semantic parameter discovery for initialization and parameter selection."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

from torch import Tensor, nn

from sequifier.config.initialization_config import LayerGroup, ParameterKind
from sequifier.model.ingestions import (
    RealFeatureProjection,
    TemporalConvFeatureIngestion,
)
from sequifier.model.layers import FeedForward, RMSNorm, SelfAttention
from sequifier.model.module_roles import (
    decoder_modules,
    decoder_output_ids,
    ingestion_projection_ids,
    position_embedding_ids,
    range_position_projection,
)

CONVOLUTIONS = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
NORMALIZATIONS = (
    nn.LayerNorm,
    RMSNorm,
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
)


@dataclass(frozen=True)
class ParameterBatch:
    group: LayerGroup
    kind: ParameterKind
    parameters: tuple[Tensor | None, ...]
    default: str
    padding_idx: int | None = None


def _module_batches(model: nn.Module) -> Iterator[ParameterBatch]:
    """Visit structural roles in initialization order, keeping joint fan groups intact."""
    modules = tuple(model.modules())
    classified = set()

    def affine(group, *layers, default="xavier_uniform"):
        classified.update(layers)
        yield ParameterBatch(
            group, "weight", tuple(layer.weight for layer in layers), default
        )
        yield ParameterBatch(
            group, "bias", tuple(layer.bias for layer in layers), "zeros"
        )

    for module in modules:
        if isinstance(module, SelfAttention):
            yield from affine(
                "attention.qkv", module.wq, module.wk, module.wv, default="joint_xavier"
            )
            if isinstance(module.wo, nn.Linear):
                yield from affine("attention.output", module.wo)
        elif isinstance(module, FeedForward):
            if module.activation == "swiglu":
                yield from affine(
                    "feed_forward.input", module.w1, module.w2, default="joint_xavier"
                )
                yield from affine("feed_forward.output", module.w3)
            else:
                yield from affine("feed_forward.input", module.linear1)
                yield from affine("feed_forward.output", module.linear2)

    for decoder in decoder_modules(model):
        outputs = decoder_output_ids(decoder)
        for layer in decoder.modules():
            if isinstance(layer, nn.Linear) and layer not in classified:
                yield from affine(
                    "decoder.output" if id(layer) in outputs else "decoder.hidden",
                    layer,
                )

    for module in modules:
        projection = range_position_projection(module)
        if projection is None:
            continue
        if not isinstance(projection, nn.Linear):
            raise TypeError("range_position_projection must be an nn.Linear")
        linear = cast(nn.Linear, projection)
        if linear in classified:
            continue
        if linear.out_features + 1 != linear.in_features:
            raise ValueError(
                "range_position_projection must map model_dim + 1 to model_dim"
            )
        yield from affine(
            "position.range_projection", linear, default="identity_plus_normal"
        )

    for module in modules:
        if isinstance(module, RealFeatureProjection):
            yield from affine("real_feature_projection", module)
    for module in modules:
        if isinstance(module, TemporalConvFeatureIngestion):
            for layer in module.layers:
                if isinstance(layer, CONVOLUTIONS):
                    yield from affine("temporal_convolution", layer)

    for module in modules:
        if isinstance(module, nn.MultiheadAttention):
            weights = (
                (module.in_proj_weight,)
                if module.in_proj_weight is not None
                else (module.q_proj_weight, module.k_proj_weight, module.v_proj_weight)
            )
            yield ParameterBatch(
                "attention.qkv",
                "weight",
                weights,
                "xavier_uniform" if len(weights) == 1 else "joint_xavier",
            )
            yield ParameterBatch(
                "attention.qkv",
                "bias",
                (module.in_proj_bias, module.bias_k, module.bias_v),
                "attention_bias",
            )
            if module.out_proj not in classified:
                yield from affine("attention.output", module.out_proj)

    positions = position_embedding_ids(model)
    projections = ingestion_projection_ids(model)
    for module in modules:
        if isinstance(module, nn.Embedding):
            group = (
                "embedding.position" if id(module) in positions else "embedding.input"
            )
            yield ParameterBatch(
                group, "weight", (module.weight,), "normal", module.padding_idx
            )
        elif isinstance(module, NORMALIZATIONS):
            yield from affine("normalization", module, default="ones")
        elif module not in classified:
            if isinstance(module, nn.Linear):
                group = (
                    "ingestion.output_projection"
                    if id(module) in projections
                    else "fallback.linear"
                )
                yield from affine(group, module)
            elif isinstance(module, CONVOLUTIONS):
                yield from affine("fallback.convolution", module)


def semantic_batches(model: nn.Module) -> Iterator[ParameterBatch]:
    seen = set()
    for batch in _module_batches(model):
        seen.update(id(p) for p in batch.parameters if p is not None)
        yield batch
    for parameter in model.parameters():
        if id(parameter) not in seen:
            yield ParameterBatch("free_parameter", "weight", (parameter,), "normal")
