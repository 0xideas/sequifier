"""Discover model parameters by Sequifier's semantic layer groups."""

from collections import defaultdict
from collections.abc import Iterable
from typing import Optional

from torch import Tensor, nn

from sequifier.config.layer_groups import LayerGroup
from sequifier.model.ingestions import (
    RealFeatureProjection,
    TemporalConvFeatureIngestion,
)
from sequifier.model.layers import FeedForward, RMSNorm, SelfAttention
from sequifier.model.module_roles import (
    decoder_module,
    decoder_output_ids,
    ingestion_projection_ids,
    position_embedding_ids,
    range_position_projection,
)
from sequifier.typechecking import beartype

CONVOLUTION_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


class SemanticParameterGroups:
    """Build a deduplicated parameter inventory using initialization precedence."""

    @beartype
    def __init__(self, model: nn.Module):
        self.model = model
        self._claimed_parameter_ids: set[int] = set()
        self._groups: dict[LayerGroup, list[nn.Parameter]] = defaultdict(list)
        self._position_embedding_module_ids = position_embedding_ids(model)
        self._ingestion_projection_module_ids = ingestion_projection_ids(model)
        self._decoder_output_module_ids = decoder_output_ids(decoder_module(model))

    @beartype
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

    @beartype
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

    @beartype
    def _claim_linear(self, layer: nn.Linear, group: LayerGroup) -> None:
        self._claim(group, (layer.weight, layer.bias))

    @beartype
    def _claim_convolution(
        self, layer: nn.Conv1d | nn.Conv2d | nn.Conv3d, group: LayerGroup
    ) -> None:
        self._claim(
            group,
            (layer.weight, layer.bias),
        )

    @beartype
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
                if module.activation == "swiglu":
                    self._claim_linear(module.w1, "feed_forward.input")
                    self._claim_linear(module.w2, "feed_forward.input")
                    self._claim_linear(module.w3, "feed_forward.output")
                else:
                    self._claim_linear(module.linear1, "feed_forward.input")
                    self._claim_linear(module.linear2, "feed_forward.output")

    @beartype
    def _collect_decoder(self) -> None:
        decoder = decoder_module(self.model)
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

    @beartype
    def _collect_range_position_projection(self) -> None:
        projection = range_position_projection(self.model)
        if isinstance(projection, nn.Linear):
            self._claim_linear(projection, "position.range_projection")

    @beartype
    def _collect_real_feature_projections(self) -> None:
        for module in self.model.modules():
            if isinstance(module, RealFeatureProjection):
                self._claim_linear(module, "real_feature_projection")

    @beartype
    def _collect_temporal_convolutions(self) -> None:
        for module in self.model.modules():
            if not isinstance(module, TemporalConvFeatureIngestion):
                continue
            for layer in module.layers:
                if isinstance(layer, CONVOLUTION_TYPES):
                    self._claim_convolution(layer, "temporal_convolution")

    @beartype
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

    @beartype
    def _collect_remaining_modules(self) -> None:
        for module in self.model.modules():
            if isinstance(module, nn.Embedding):
                group: LayerGroup = (
                    "embedding.position"
                    if id(module) in self._position_embedding_module_ids
                    else "embedding.input"
                )
                self._claim(group, (module.weight,))
            elif isinstance(
                module,
                (
                    nn.LayerNorm,
                    RMSNorm,
                    nn.GroupNorm,
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                ),
            ):
                self._claim(
                    "normalization",
                    (module.weight, module.bias),
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

    @beartype
    def _collect_free_parameters(self) -> None:
        for _, parameter in self.model.named_parameters(remove_duplicate=True):
            self._claim("free_parameter", (parameter,))


@beartype
def semantic_parameter_groups(
    model: nn.Module,
) -> dict[LayerGroup, tuple[nn.Parameter, ...]]:
    """Return the semantic parameter inventory for ``model``."""

    return SemanticParameterGroups(model).collect()
