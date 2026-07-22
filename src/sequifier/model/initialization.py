"""Centralized trainable-parameter initialization for Sequifier models."""

import math
from collections.abc import Iterable
from typing import Optional, cast

import torch
from torch import Tensor, nn

from sequifier.model.ingestions import (
    RealFeatureProjection,
    TemporalConvFeatureIngestion,
)
from sequifier.model.layers import FeedForward, RMSNorm, SelfAttention

EMBEDDING_INIT_STD = 0.02
DECODER_REFERENCE_STD = 0.02
CONVOLUTION_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
Convolution = nn.Conv1d | nn.Conv2d | nn.Conv3d


def _fan_in(weight: Tensor) -> int:
    if weight.ndim < 2:
        raise ValueError(
            f"Fan-in initialization requires at least two dimensions, got {weight.shape}."
        )
    return weight.shape[1] * math.prod(weight.shape[2:])


def _fan_out(weight: Tensor) -> int:
    if weight.ndim < 2:
        raise ValueError(
            f"Fan-out initialization requires at least two dimensions, got {weight.shape}."
        )
    return weight.shape[0] * math.prod(weight.shape[2:])


class _ModelWeightInitializer:
    """Apply one initialization policy while visiting every parameter once."""

    def __init__(
        self,
        model: nn.Module,
        *,
        transformer_depth: int,
        decoder_reference_dim: int,
    ):
        if transformer_depth <= 0:
            raise ValueError("transformer_depth must be positive")
        if decoder_reference_dim <= 0:
            raise ValueError("decoder_reference_dim must be positive")

        self.model = model
        self.transformer_depth = transformer_depth
        self.decoder_reference_dim = decoder_reference_dim
        self.residual_scale = 1.0 / math.sqrt(2.0 * transformer_depth)
        self.initialized_parameter_ids: set[int] = set()

    def initialize(self) -> None:
        self._initialize_transformer_layers()
        self._initialize_decoder()
        self._initialize_range_position_projection()
        self._initialize_real_feature_projections()
        self._initialize_temporal_convolutions()
        self._initialize_multihead_attention()
        self._initialize_remaining_modules()
        self._initialize_free_parameters()

    def _is_initialized(self, parameter: Optional[Tensor]) -> bool:
        return parameter is None or id(parameter) in self.initialized_parameter_ids

    def _mark_initialized(self, parameter: Optional[Tensor]) -> None:
        if parameter is not None:
            self.initialized_parameter_ids.add(id(parameter))

    def _zero_(self, parameter: Optional[Tensor]) -> None:
        if parameter is None or self._is_initialized(parameter):
            return
        nn.init.zeros_(parameter)
        self._mark_initialized(parameter)

    def _normal_(self, parameter: Tensor, std: float) -> None:
        if self._is_initialized(parameter):
            return
        nn.init.normal_(parameter, mean=0.0, std=std)
        self._mark_initialized(parameter)

    def _xavier_uniform_(self, parameter: Tensor, gain: float = 1.0) -> None:
        if self._is_initialized(parameter):
            return
        nn.init.xavier_uniform_(parameter, gain=gain)
        self._mark_initialized(parameter)

    def _joint_xavier_uniform_(
        self,
        parameters: Iterable[Tensor],
        *,
        gain: float = 1.0,
    ) -> None:
        weights = list(parameters)
        joint_fan_out = sum(_fan_out(weight) for weight in weights)
        for weight in weights:
            if self._is_initialized(weight):
                continue
            bound = gain * math.sqrt(6.0 / (_fan_in(weight) + joint_fan_out))
            nn.init.uniform_(weight, -bound, bound)
            self._mark_initialized(weight)

    def _initialize_linear_xavier(
        self,
        layer: nn.Linear,
        *,
        residual: bool = False,
    ) -> None:
        if not self._is_initialized(layer.weight):
            nn.init.xavier_uniform_(layer.weight)
            if residual:
                with torch.no_grad():
                    layer.weight.mul_(self.residual_scale)
            self._mark_initialized(layer.weight)
        self._zero_(layer.bias)

    def _initialize_linear_kaiming_relu(self, layer: nn.Linear) -> None:
        if not self._is_initialized(layer.weight):
            nn.init.kaiming_uniform_(
                layer.weight,
                mode="fan_in",
                nonlinearity="relu",
            )
            self._mark_initialized(layer.weight)
        self._zero_(layer.bias)

    def _initialize_linear_fan_in(self, layer: nn.Linear) -> None:
        if not self._is_initialized(layer.weight):
            nn.init.kaiming_uniform_(
                layer.weight,
                mode="fan_in",
                nonlinearity="linear",
            )
            self._mark_initialized(layer.weight)
        self._zero_(layer.bias)

    def _initialize_conv_xavier(self, layer: Convolution) -> None:
        self._xavier_uniform_(layer.weight)
        self._zero_(layer.bias)

    def _initialize_conv_kaiming_relu(self, layer: Convolution) -> None:
        if not self._is_initialized(layer.weight):
            nn.init.kaiming_uniform_(
                layer.weight,
                mode="fan_in",
                nonlinearity="relu",
            )
            self._mark_initialized(layer.weight)
        self._zero_(layer.bias)

    def _initialize_transformer_layers(self) -> None:
        for module in self.model.modules():
            if isinstance(module, SelfAttention):
                qkv_weights = [
                    module.wq.weight,
                    module.wk.weight,
                    module.wv.weight,
                ]
                self._joint_xavier_uniform_(qkv_weights)
                self._zero_(module.wq.bias)
                self._zero_(module.wk.bias)
                self._zero_(module.wv.bias)
                self._initialize_linear_xavier(module.wo, residual=True)
            elif isinstance(module, FeedForward):
                self._initialize_feed_forward(module)

    def _initialize_feed_forward(self, feed_forward: FeedForward) -> None:
        if feed_forward.activation_fn == "swiglu":
            self._joint_xavier_uniform_(
                [feed_forward.w1.weight, feed_forward.w2.weight]
            )
            self._zero_(feed_forward.w1.bias)
            self._zero_(feed_forward.w2.bias)
            self._initialize_linear_xavier(feed_forward.w3, residual=True)
            return

        if feed_forward.activation_fn == "relu":
            self._initialize_linear_kaiming_relu(feed_forward.linear1)
        else:
            self._initialize_linear_xavier(feed_forward.linear1)
        self._initialize_linear_xavier(feed_forward.linear2, residual=True)

    def _initialize_decoder(self) -> None:
        decoder = getattr(self.model, "decoder", None)
        if not isinstance(decoder, nn.Module):
            return

        for module in cast(nn.Module, decoder).modules():
            if not isinstance(module, nn.Linear):
                continue
            std = DECODER_REFERENCE_STD * math.sqrt(
                self.decoder_reference_dim / module.in_features
            )
            self._normal_(module.weight, std)
            self._zero_(module.bias)

    def _initialize_range_position_projection(self) -> None:
        projection = getattr(self.model, "range_position_projection", None)
        if projection is None:
            return
        if not isinstance(projection, nn.Linear):
            raise TypeError("range_position_projection must be an nn.Linear")
        if projection.out_features + 1 != projection.in_features:
            raise ValueError(
                "range_position_projection must map dim_model + 1 to dim_model"
            )

        if not self._is_initialized(projection.weight):
            with torch.no_grad():
                projection.weight.zero_()
                projection.weight[:, : projection.out_features].copy_(
                    torch.eye(
                        projection.out_features,
                        device=projection.weight.device,
                        dtype=projection.weight.dtype,
                    )
                )
                projection.weight[:, -1].normal_(
                    mean=0.0,
                    std=EMBEDDING_INIT_STD,
                )
            self._mark_initialized(projection.weight)
        self._zero_(projection.bias)

    def _initialize_real_feature_projections(self) -> None:
        for module in self.model.modules():
            if isinstance(module, RealFeatureProjection):
                self._initialize_linear_fan_in(module)

    def _initialize_temporal_convolutions(self) -> None:
        for module in self.model.modules():
            if not isinstance(module, TemporalConvFeatureIngestion):
                continue
            for layer in module.layers:
                if not isinstance(layer, CONVOLUTION_TYPES):
                    continue
                if isinstance(module.activation, nn.ReLU):
                    self._initialize_conv_kaiming_relu(layer)
                else:
                    self._initialize_conv_xavier(layer)

    def _initialize_multihead_attention(self) -> None:
        for module in self.model.modules():
            if not isinstance(module, nn.MultiheadAttention):
                continue

            if module.in_proj_weight is not None:
                self._xavier_uniform_(module.in_proj_weight)
            else:
                qkv_weights = [
                    weight
                    for weight in (
                        module.q_proj_weight,
                        module.k_proj_weight,
                        module.v_proj_weight,
                    )
                    if weight is not None
                ]
                self._joint_xavier_uniform_(qkv_weights)

            self._zero_(module.in_proj_bias)
            self._initialize_linear_xavier(module.out_proj)

            if module.bias_k is not None and not self._is_initialized(module.bias_k):
                nn.init.xavier_normal_(module.bias_k)
                self._mark_initialized(module.bias_k)
            if module.bias_v is not None and not self._is_initialized(module.bias_v):
                nn.init.xavier_normal_(module.bias_v)
                self._mark_initialized(module.bias_v)

    def _initialize_remaining_modules(self) -> None:
        for module in self.model.modules():
            if isinstance(module, nn.Embedding):
                self._normal_(module.weight, EMBEDDING_INIT_STD)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
            elif isinstance(module, (nn.LayerNorm, RMSNorm)):
                weight = getattr(module, "weight", None)
                if isinstance(weight, Tensor) and not self._is_initialized(weight):
                    nn.init.ones_(weight)
                    self._mark_initialized(weight)
                self._zero_(getattr(module, "bias", None))
            elif isinstance(module, nn.Linear):
                self._initialize_linear_xavier(module)
            elif isinstance(module, CONVOLUTION_TYPES):
                self._initialize_conv_xavier(module)

    def _initialize_free_parameters(self) -> None:
        for _, parameter in self.model.named_parameters(remove_duplicate=True):
            if self._is_initialized(parameter):
                continue
            if not parameter.is_floating_point():
                raise TypeError(
                    "Sequifier only supports floating-point trainable parameters, "
                    f"got {parameter.dtype}."
                )
            self._normal_(parameter, EMBEDDING_INIT_STD)


def initialize_model_weights(
    model: nn.Module,
    *,
    transformer_depth: int,
    decoder_reference_dim: int,
) -> None:
    """Initialize all model parameters according to Sequifier's unified policy.

    Q/K/V projections use Glorot uniform with the logical packed projection's
    joint fan-out. Transformer residual projections are additionally scaled by
    ``1 / sqrt(2 * transformer_depth)``. ReLU paths use Kaiming initialization;
    scalar real-feature projections use fan-in-preserving linear initialization,
    and other linear and convolutional paths use Glorot. Embeddings and free
    learned vectors use a small normal distribution. Decoder weights retain a
    small initial output scale while compensating for their configured fan-in.
    """

    _ModelWeightInitializer(
        model,
        transformer_depth=transformer_depth,
        decoder_reference_dim=decoder_reference_dim,
    ).initialize()
