"""Structural module roles shared by initialization and parameter selection."""

from torch import nn

from sequifier.model.decoders import (
    AutoregressiveTransformerDecoderBranch,
    HashTargetHead,
    TargetDecoderBranch,
    TargetDecoding,
)
from sequifier.model.ingestions import IngestionOutputProjection

_POSITION_MODULES = {
    "pos_encoder",
    "position_embedding",
    "global_position_encoder",
    "axis_embedding_layers",
}
_INGESTION_PROJECTION_MODULES = {
    "output_projection_layer",
    "input_projection",
    "concat_projection",
    "branch_projections",
}


def decoder_module(model: nn.Module) -> nn.Module | None:
    if isinstance(model, (TargetDecoding, TargetDecoderBranch)):
        return model
    try:
        return model.get_submodule("decoder")
    except AttributeError:
        return None


def position_embedding_ids(model: nn.Module) -> set[int]:
    return {
        id(module)
        for name, module in model.named_modules(remove_duplicate=False)
        if isinstance(module, nn.Embedding)
        and _POSITION_MODULES.intersection(name.split("."))
    }


def ingestion_projection_ids(model: nn.Module) -> set[int]:
    return {
        id(module)
        for name, module in model.named_modules(remove_duplicate=False)
        if isinstance(module, nn.Linear)
        and (
            isinstance(module, IngestionOutputProjection)
            or _INGESTION_PROJECTION_MODULES.intersection(name.split("."))
        )
    }


def decoder_output_ids(decoder: nn.Module | None) -> set[int]:
    if decoder is None:
        return set()
    ids = {
        id(output)
        for branch in decoder.modules()
        if isinstance(
            branch, (TargetDecoderBranch, AutoregressiveTransformerDecoderBranch)
        )
        for output in branch.output_layers.values()
        if isinstance(output, nn.Linear)
    }
    ids.update(
        id(layer)
        for head in decoder.modules()
        if isinstance(head, HashTargetHead)
        for layer in head.layers
    )
    # Legacy decoders were per-target ModuleDicts.
    return ids or {
        id(module) for module in decoder.modules() if isinstance(module, nn.Linear)
    }


def range_position_projection(model: nn.Module) -> object:
    # Accept both historical names for standalone backbones.
    try:
        projection = model.range_position_projection
    except AttributeError:
        projection = None
    if projection is not None:
        return projection
    try:
        return model.range_projection
    except AttributeError:
        return None
