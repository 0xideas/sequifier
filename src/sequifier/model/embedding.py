import re
from dataclasses import dataclass
from typing import Any, Literal

from sequifier.typechecking import beartype

DEFAULT_EMBEDDING_LAYER_NAMES = ["backbone.final_norm"]
ONNX_EMBEDDING_LAYER_NAMES_KEY = "sequifier.embedding_layer_names"

_BACKBONE_LAYER_PATTERN = re.compile(r"backbone\.layers\.(?P<index>\d+)")
_DECODER_HIDDEN_BLOCK_PATTERN = re.compile(
    r"decoder\.branches\.(?P<branch>[^.]+)\.hidden_blocks\.(?P<index>\d+)"
)


@dataclass(frozen=True)
class EmbeddingLayerSelector:
    """Parsed, stable selector for one embedding activation source."""

    name: str
    source: Literal["backbone_layer", "backbone_final_norm", "decoder_hidden_block"]
    index: int | None = None
    branch: str | None = None


@beartype
def parse_embedding_layer_name(name: str) -> EmbeddingLayerSelector:
    """Parse one public embedding-layer name."""
    if name == "backbone.final_norm":
        return EmbeddingLayerSelector(name=name, source="backbone_final_norm")

    backbone_match = _BACKBONE_LAYER_PATTERN.fullmatch(name)
    if backbone_match is not None:
        return EmbeddingLayerSelector(
            name=name,
            source="backbone_layer",
            index=int(backbone_match.group("index")),
        )

    decoder_match = _DECODER_HIDDEN_BLOCK_PATTERN.fullmatch(name)
    if decoder_match is not None:
        return EmbeddingLayerSelector(
            name=name,
            source="decoder_hidden_block",
            branch=decoder_match.group("branch"),
            index=int(decoder_match.group("index")),
        )

    raise ValueError(
        f"Unknown embedding layer name {name!r}. Expected 'backbone.final_norm', "
        "'backbone.layers.<index>', or "
        "'decoder.branches.<branch>.hidden_blocks.<index>'."
    )


@beartype
def available_embedding_layer_names(model_spec: Any) -> list[str]:
    """Return every valid embedding activation selector for a model spec."""
    architecture = model_spec.backbone.architecture
    names = [
        *(f"backbone.layers.{index}" for index in range(architecture.num_layers)),
        "backbone.final_norm",
    ]

    decoder = model_spec.decoder
    branch_items = (
        decoder.branches.items()
        if decoder.type == "composite"
        else (("default", decoder),)
    )
    for branch_name, branch in branch_items:
        if branch.type != "mlp":
            continue
        names.extend(
            f"decoder.branches.{branch_name}.hidden_blocks.{index}"
            for index in range(len(branch.hidden_dims))
        )
    return names


@beartype
def validate_embedding_layer_names(
    layer_names: list[str], model_spec: Any
) -> list[str]:
    """Validate configured embedding selectors against the concrete model."""
    if len(layer_names) != len(set(layer_names)):
        raise ValueError("embedding_layer_names cannot contain duplicates")

    allowed_names = available_embedding_layer_names(model_spec)
    allowed = set(allowed_names)
    invalid_names = [name for name in layer_names if name not in allowed]
    if invalid_names:
        raise ValueError(
            f"Unknown embedding_layer_names {invalid_names}. "
            f"Available names are {allowed_names}."
        )
    return layer_names


@beartype
def embedding_layer_trace_site(name: str) -> str:
    selector = parse_embedding_layer_name(name)
    if selector.source == "backbone_final_norm":
        return "backbone.final_norm"
    if selector.source == "backbone_layer":
        return f"backbone.layer.{selector.index}.output"
    return f"decoder.branch.{selector.branch}.block.{selector.index}"
