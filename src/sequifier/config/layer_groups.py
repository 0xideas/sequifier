"""Semantic model-parameter groups shared by parameter policies."""

from typing import Literal, TypeAlias

LayerGroup: TypeAlias = Literal[
    "embedding.input",
    "embedding.position",
    "ingestion.output_projection",
    "real_feature_projection",
    "temporal_convolution",
    "attention.qkv",
    "attention.output",
    "feed_forward.input",
    "feed_forward.output",
    "decoder.hidden",
    "decoder.output",
    "normalization",
    "position.range_projection",
    "fallback.linear",
    "fallback.convolution",
    "free_parameter",
]
