"""Discover model parameters by Sequifier's semantic layer groups."""

from collections import defaultdict
from collections.abc import Iterable
from typing import Optional, cast

from torch import Tensor, nn

from sequifier.config.layer_groups import LayerGroup
from sequifier.model.decoders import TargetDecoderBranch, TargetDecoding
from sequifier.model.ingestions import (
    RealFeatureProjection,
    TemporalConvFeatureIngestion,
)
from sequifier.model.layers import FeedForward, RMSNorm, SelfAttention

CONVOLUTION_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


class SemanticParameterGroups:
    """Build a deduplicated parameter inventory using initialization precedence."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._claimed_parameter_ids: set[int] = set()
        self._groups: dict[LayerGroup, list[nn.Parameter]] = defaultdict(list)
        self._position_embedding_module_ids = self._position_embedding_ids()
        self._ingestion_projection_module_ids = self._ingestion_projection_ids()
        self._decoder_output_module_ids = self._decoder_output_ids()

    def collect(self) -> dict[LayerGroup, tuple[nn.Parameter, ...]]:
        """Return every parameter exactly once under its semantic group."""

        self._collect_transformer_layers()
        self._collect_decoder()
        self._collect_range_position_projection()
        self._collect_real_feature_projections()
        self._collect_temporal_convolutions()
        self._collect_multihead_attention()
        self._collect_remaining_modules()
        self._collect_free_parameters()
        return {group: tuple(parameters) for group, parameters in self._groups.items()}

    def _claim(
        self,
        group: LayerGroup,
        parameters: Iterable[Optional[Tensor]],
    ) -> None:
        for parameter in parameters:
            if not isinstance(parameter, nn.Parameter):
                continue
            parameter_id = id(parameter)
            if parameter_id in self._claimed_parameter_ids:
                continue
            self._claimed_parameter_ids.add(parameter_id)
            self._groups[group].append(parameter)

    def _decoder_module(self) -> Optional[nn.Module]:
        if isinstance(self.model, (TargetDecoding, TargetDecoderBranch)):
            return self.model
        decoder = getattr(self.model, "decoder", None)
        return decoder if isinstance(decoder, nn.Module) else None

    def _position_embedding_ids(self) -> set[int]:
        position_embeddings: set[int] = set()
        for module in self.model.modules():
            for attribute in (
                "pos_encoder",
                "position_embedding",
                "global_position_encoder",
                "axis_embedding_layers",
            ):
                value = getattr(module, attribute, None)
                if not isinstance(value, nn.Module):
                    continue
                position_embeddings.update(
                    id(child)
                    for child in cast(nn.Module, value).modules()
                    if isinstance(child, nn.Embedding)
                )
        return position_embeddings

    def _ingestion_projection_ids(self) -> set[int]:
        projections: set[int] = set()
        for module in self.model.modules():
            if (
                isinstance(module, nn.Linear)
                and getattr(module, "_sequifier_layer_group", None)
                == "ingestion.output_projection"
            ):
                projections.add(id(module))
            for attribute in (
                "output_projection_layer",
                "input_projection",
                "concat_projection",
                "branch_projections",
            ):
                value = getattr(module, attribute, None)
                if not isinstance(value, nn.Module):
                    continue
                projections.update(
                    id(child)
                    for child in cast(nn.Module, value).modules()
                    if isinstance(child, nn.Linear)
                )
        return projections

    def _decoder_output_ids(self) -> set[int]:
        decoder = self._decoder_module()
        if decoder is None:
            return set()
        output_module_ids = {
            id(output_layer)
            for module in decoder.modules()
            if isinstance(module, TargetDecoderBranch)
            for output_layer in module.output_layers.values()
            if isinstance(output_layer, nn.Linear)
        }
        if output_module_ids:
            return output_module_ids
        return {
            id(module) for module in decoder.modules() if isinstance(module, nn.Linear)
        }

    def _claim_linear(self, layer: nn.Linear, group: LayerGroup) -> None:
        self._claim(group, (layer.weight, layer.bias))

    def _claim_convolution(self, layer: nn.Module, group: LayerGroup) -> None:
        self._claim(
            group,
            (getattr(layer, "weight", None), getattr(layer, "bias", None)),
        )

    def _collect_transformer_layers(self) -> None:
        for module in self.model.modules():
            if isinstance(module, SelfAttention):
                self._claim(
                    "attention.qkv",
                    (
                        module.wq.weight,
                        module.wq.bias,
                        module.wk.weight,
                        module.wk.bias,
                        module.wv.weight,
                        module.wv.bias,
                    ),
                )
                if isinstance(module.wo, nn.Linear):
                    self._claim_linear(module.wo, "attention.output")
            elif isinstance(module, FeedForward):
                if module.activation_fn == "swiglu":
                    self._claim_linear(module.w1, "feed_forward.input")
                    self._claim_linear(module.w2, "feed_forward.input")
                    self._claim_linear(module.w3, "feed_forward.output")
                else:
                    self._claim_linear(module.linear1, "feed_forward.input")
                    self._claim_linear(module.linear2, "feed_forward.output")

    def _collect_decoder(self) -> None:
        decoder = self._decoder_module()
        if decoder is None:
            return
        for module in decoder.modules():
            if not isinstance(module, nn.Linear):
                continue
            group: LayerGroup = (
                "decoder.output"
                if id(module) in self._decoder_output_module_ids
                else "decoder.hidden"
            )
            self._claim_linear(module, group)

    def _collect_range_position_projection(self) -> None:
        projection = getattr(self.model, "range_position_projection", None)
        if projection is None:
            projection = getattr(self.model, "range_projection", None)
        if isinstance(projection, nn.Linear):
            self._claim_linear(projection, "position.range_projection")

    def _collect_real_feature_projections(self) -> None:
        for module in self.model.modules():
            if isinstance(module, RealFeatureProjection):
                self._claim_linear(module, "real_feature_projection")

    def _collect_temporal_convolutions(self) -> None:
        for module in self.model.modules():
            if not isinstance(module, TemporalConvFeatureIngestion):
                continue
            for layer in module.layers:
                if isinstance(layer, CONVOLUTION_TYPES):
                    self._claim_convolution(layer, "temporal_convolution")

    def _collect_multihead_attention(self) -> None:
        for module in self.model.modules():
            if not isinstance(module, nn.MultiheadAttention):
                continue
            self._claim(
                "attention.qkv",
                (
                    module.in_proj_weight,
                    module.q_proj_weight,
                    module.k_proj_weight,
                    module.v_proj_weight,
                    module.in_proj_bias,
                    module.bias_k,
                    module.bias_v,
                ),
            )
            self._claim_linear(module.out_proj, "attention.output")

    def _collect_remaining_modules(self) -> None:
        for module in self.model.modules():
            if isinstance(module, nn.Embedding):
                group: LayerGroup = (
                    "embedding.position"
                    if id(module) in self._position_embedding_module_ids
                    else "embedding.input"
                )
                self._claim(group, (module.weight,))
            elif isinstance(module, (nn.LayerNorm, RMSNorm)):
                self._claim(
                    "normalization",
                    (getattr(module, "weight", None), getattr(module, "bias", None)),
                )
            elif isinstance(module, nn.Linear):
                group = (
                    "ingestion.output_projection"
                    if id(module) in self._ingestion_projection_module_ids
                    else "fallback.linear"
                )
                self._claim_linear(module, group)
            elif isinstance(module, CONVOLUTION_TYPES):
                self._claim_convolution(module, "fallback.convolution")

    def _collect_free_parameters(self) -> None:
        for _, parameter in self.model.named_parameters(remove_duplicate=True):
            self._claim("free_parameter", (parameter,))


def semantic_parameter_groups(
    model: nn.Module,
) -> dict[LayerGroup, tuple[nn.Parameter, ...]]:
    """Return the semantic parameter inventory for ``model``."""

    return SemanticParameterGroups(model).collect()
