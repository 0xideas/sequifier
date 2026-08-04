"""Centralized, configurable trainable-parameter initialization."""

import math
from collections.abc import Iterable
from typing import Optional, TypeAlias, cast

import torch
from torch import Tensor, nn

from sequifier.config.initialization_config import (
    InitializationMethodConfig,
    LayerGroup,
    ModelInitializationConfig,
)
from sequifier.model.decoders import TargetDecoderBranch, TargetDecoding
from sequifier.model.ingestions import (
    RealFeatureProjection,
    TemporalConvFeatureIngestion,
)
from sequifier.model.layers import FeedForward, RMSNorm, SelfAttention

EMBEDDING_INIT_STD = 0.02
CONVOLUTION_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
Convolution: TypeAlias = nn.Conv1d | nn.Conv2d | nn.Conv3d


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
    """Apply current defaults with optional semantic layer-group overrides."""

    def __init__(
        self,
        model: nn.Module,
        initialization: Optional[ModelInitializationConfig],
    ):
        self.model = model
        self.initialization = initialization or ModelInitializationConfig()
        self.initialized_parameter_ids: set[int] = set()
        self.position_embedding_module_ids = self._position_embedding_module_ids()
        self.ingestion_projection_module_ids = self._ingestion_projection_module_ids()
        self.decoder_output_module_ids = self._decoder_output_module_ids()

    def _decoder_module(self) -> Optional[nn.Module]:
        if isinstance(self.model, (TargetDecoding, TargetDecoderBranch)):
            return self.model
        decoder = getattr(self.model, "decoder", None)
        return decoder if isinstance(decoder, nn.Module) else None

    def initialize(self) -> None:
        # Keep this order stable: it is part of seeded initialization compatibility.
        self._initialize_transformer_layers()
        self._initialize_decoder()
        self._initialize_range_position_projection()
        self._initialize_real_feature_projections()
        self._initialize_temporal_convolutions()
        self._initialize_multihead_attention()
        self._initialize_remaining_modules()
        self._initialize_free_parameters()

    def _position_embedding_module_ids(self) -> set[int]:
        position_embeddings: set[int] = set()
        for module in self.model.modules():
            for attribute in (
                "pos_encoder",
                "global_position_encoder",
                "axis_embedding_layers",
            ):
                value = getattr(module, attribute, None)
                if not isinstance(value, nn.Module):
                    continue
                value_module = cast(nn.Module, value)
                position_embeddings.update(
                    id(child)
                    for child in value_module.modules()
                    if isinstance(child, nn.Embedding)
                )
        return position_embeddings

    def _ingestion_projection_module_ids(self) -> set[int]:
        projections: set[int] = set()
        for module in self.model.modules():
            for attribute in (
                "output_projection_layer",
                "input_projection",
                "concat_projection",
                "branch_projections",
            ):
                value = getattr(module, attribute, None)
                if not isinstance(value, nn.Module):
                    continue
                value_module = cast(nn.Module, value)
                projections.update(
                    id(child)
                    for child in value_module.modules()
                    if isinstance(child, nn.Linear)
                )
        return projections

    def _decoder_output_module_ids(self) -> set[int]:
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
        # Compatibility with the original per-target ModuleDict decoder.
        return {
            id(module) for module in decoder.modules() if isinstance(module, nn.Linear)
        }

    def _is_initialized(self, parameter: Optional[Tensor]) -> bool:
        return parameter is None or id(parameter) in self.initialized_parameter_ids

    def _mark_initialized(self, parameter: Optional[Tensor]) -> None:
        if parameter is not None:
            self.initialized_parameter_ids.add(id(parameter))

    def _override_method(
        self,
        group: LayerGroup,
        parameter_kind: str,
    ) -> Optional[InitializationMethodConfig]:
        override = self.initialization.override_for(group)
        if override is None:
            return None
        if parameter_kind == "weight":
            return override.weight
        if parameter_kind == "bias":
            return override.bias
        raise ValueError(f"Unknown parameter kind: {parameter_kind}")

    def _apply_override(
        self,
        group: LayerGroup,
        parameter_kind: str,
        parameters: Iterable[Optional[Tensor]],
    ) -> bool:
        method = self._override_method(group, parameter_kind)
        if method is None:
            return False
        self._apply_method(group, parameter_kind, parameters, method)
        return True

    def _apply_method(
        self,
        group: LayerGroup,
        parameter_kind: str,
        parameters: Iterable[Optional[Tensor]],
        config: InitializationMethodConfig,
    ) -> None:
        tensors = [
            parameter
            for parameter in parameters
            if parameter is not None and not self._is_initialized(parameter)
        ]
        if not tensors:
            return

        method = config.method
        if method == "preserve":
            for parameter in tensors:
                self._mark_initialized(parameter)
            return

        for parameter in tensors:
            if not parameter.is_floating_point():
                raise TypeError(
                    f"Initialization for {group}.{parameter_kind} requires a "
                    f"floating-point tensor, got {parameter.dtype}."
                )

        try:
            if method == "normal":
                for parameter in tensors:
                    nn.init.normal_(parameter, mean=config.mean, std=config.std)
            elif method == "uniform":
                for parameter in tensors:
                    nn.init.uniform_(parameter, a=config.low, b=config.high)
            elif method in {
                "xavier",
                "glorot",
                "xavier_uniform",
                "glorot_uniform",
            }:
                if config.fan_mode == "joint":
                    self._joint_xavier_uniform_raw(tensors, gain=config.gain)
                else:
                    for parameter in tensors:
                        nn.init.xavier_uniform_(parameter, gain=config.gain)
            elif method in {"xavier_normal", "glorot_normal"}:
                if config.fan_mode == "joint":
                    self._joint_xavier_normal_raw(tensors, gain=config.gain)
                else:
                    for parameter in tensors:
                        nn.init.xavier_normal_(parameter, gain=config.gain)
            elif method == "kaiming_uniform":
                for parameter in tensors:
                    nn.init.kaiming_uniform_(
                        parameter,
                        a=config.a,
                        mode=config.mode,
                        nonlinearity=config.nonlinearity,
                    )
            elif method == "kaiming_normal":
                for parameter in tensors:
                    nn.init.kaiming_normal_(
                        parameter,
                        a=config.a,
                        mode=config.mode,
                        nonlinearity=config.nonlinearity,
                    )
            elif method == "constant":
                for parameter in tensors:
                    nn.init.constant_(parameter, config.value)
            elif method == "zeros":
                for parameter in tensors:
                    nn.init.zeros_(parameter)
            elif method == "ones":
                for parameter in tensors:
                    nn.init.ones_(parameter)
            elif method == "identity_plus_normal":
                for parameter in tensors:
                    self._identity_plus_normal_raw(
                        parameter,
                        mean=config.mean,
                        std=config.std,
                    )
            else:
                raise ValueError(f"Unknown initialization method: {method}")
        except (RuntimeError, ValueError) as error:
            shapes = [tuple(parameter.shape) for parameter in tensors]
            raise ValueError(
                f"Cannot apply {method!r} to {group}.{parameter_kind} "
                f"with tensor shapes {shapes}: {error}"
            ) from error

        for parameter in tensors:
            self._mark_initialized(parameter)

    @staticmethod
    def _joint_xavier_uniform_raw(
        parameters: Iterable[Tensor],
        *,
        gain: float = 1.0,
    ) -> None:
        weights = list(parameters)
        joint_fan_out = sum(_fan_out(weight) for weight in weights)
        for weight in weights:
            bound = gain * math.sqrt(6.0 / (_fan_in(weight) + joint_fan_out))
            nn.init.uniform_(weight, -bound, bound)

    @staticmethod
    def _joint_xavier_normal_raw(
        parameters: Iterable[Tensor],
        *,
        gain: float = 1.0,
    ) -> None:
        weights = list(parameters)
        joint_fan_out = sum(_fan_out(weight) for weight in weights)
        for weight in weights:
            std = gain * math.sqrt(2.0 / (_fan_in(weight) + joint_fan_out))
            nn.init.normal_(weight, mean=0.0, std=std)

    @staticmethod
    def _identity_plus_normal_raw(
        parameter: Tensor,
        *,
        mean: float,
        std: float,
    ) -> None:
        if parameter.ndim != 2 or parameter.shape[1] != parameter.shape[0] + 1:
            raise ValueError(
                "identity_plus_normal requires a matrix with one more input "
                f"than output feature, got {tuple(parameter.shape)}"
            )
        with torch.no_grad():
            parameter.zero_()
            parameter[:, :-1].copy_(
                torch.eye(
                    parameter.shape[0],
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )
            parameter[:, -1].normal_(mean=mean, std=std)

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
        group: LayerGroup,
    ) -> None:
        if not self._apply_override(group, "weight", [layer.weight]):
            self._xavier_uniform_(layer.weight)
        if not self._apply_override(group, "bias", [layer.bias]):
            self._zero_(layer.bias)

    def _initialize_conv_xavier(
        self,
        layer: Convolution,
        group: LayerGroup,
    ) -> None:
        if not self._apply_override(group, "weight", [layer.weight]):
            self._xavier_uniform_(layer.weight)
        if not self._apply_override(group, "bias", [layer.bias]):
            self._zero_(layer.bias)

    def _initialize_transformer_layers(self) -> None:
        for module in self.model.modules():
            if isinstance(module, SelfAttention):
                qkv_weights = [
                    module.wq.weight,
                    module.wk.weight,
                    module.wv.weight,
                ]
                if not self._apply_override("attention.qkv", "weight", qkv_weights):
                    self._joint_xavier_uniform_(qkv_weights)
                if not self._apply_override(
                    "attention.qkv",
                    "bias",
                    [module.wq.bias, module.wk.bias, module.wv.bias],
                ):
                    self._zero_(module.wq.bias)
                    self._zero_(module.wk.bias)
                    self._zero_(module.wv.bias)
                if isinstance(module.wo, nn.Linear):
                    self._initialize_linear_xavier(module.wo, "attention.output")
            elif isinstance(module, FeedForward):
                self._initialize_feed_forward(module)

    def _initialize_feed_forward(self, feed_forward: FeedForward) -> None:
        if feed_forward.activation_fn == "swiglu":
            input_weights = [feed_forward.w1.weight, feed_forward.w2.weight]
            if not self._apply_override("feed_forward.input", "weight", input_weights):
                self._joint_xavier_uniform_(input_weights)
            if not self._apply_override(
                "feed_forward.input",
                "bias",
                [feed_forward.w1.bias, feed_forward.w2.bias],
            ):
                self._zero_(feed_forward.w1.bias)
                self._zero_(feed_forward.w2.bias)
            self._initialize_linear_xavier(feed_forward.w3, "feed_forward.output")
            return

        self._initialize_linear_xavier(feed_forward.linear1, "feed_forward.input")
        self._initialize_linear_xavier(feed_forward.linear2, "feed_forward.output")

    def _initialize_decoder(self) -> None:
        decoder = self._decoder_module()
        if decoder is None:
            return

        for module in decoder.modules():
            if not isinstance(module, nn.Linear):
                continue
            group: LayerGroup = (
                "decoder.output"
                if id(module) in self.decoder_output_module_ids
                else "decoder.hidden"
            )
            self._initialize_linear_xavier(module, group)

    def _initialize_range_position_projection(self) -> None:
        projection = getattr(self.model, "range_position_projection", None)
        if projection is None:
            projection = getattr(self.model, "range_projection", None)
        if projection is None:
            return
        if not isinstance(projection, nn.Linear):
            raise TypeError("range_position_projection must be an nn.Linear")
        if projection.out_features + 1 != projection.in_features:
            raise ValueError(
                "range_position_projection must map dim_model + 1 to dim_model"
            )

        if not self._apply_override(
            "position.range_projection", "weight", [projection.weight]
        ):
            if not self._is_initialized(projection.weight):
                self._identity_plus_normal_raw(
                    projection.weight,
                    mean=0.0,
                    std=EMBEDDING_INIT_STD,
                )
                self._mark_initialized(projection.weight)
        if not self._apply_override(
            "position.range_projection", "bias", [projection.bias]
        ):
            self._zero_(projection.bias)

    def _initialize_real_feature_projections(self) -> None:
        for module in self.model.modules():
            if isinstance(module, RealFeatureProjection):
                self._initialize_linear_xavier(module, "real_feature_projection")

    def _initialize_temporal_convolutions(self) -> None:
        for module in self.model.modules():
            if not isinstance(module, TemporalConvFeatureIngestion):
                continue
            for layer in module.layers:
                if isinstance(layer, CONVOLUTION_TYPES):
                    self._initialize_conv_xavier(layer, "temporal_convolution")

    def _initialize_multihead_attention(self) -> None:
        for module in self.model.modules():
            if not isinstance(module, nn.MultiheadAttention):
                continue

            qkv_weights = (
                [module.in_proj_weight]
                if module.in_proj_weight is not None
                else [
                    weight
                    for weight in (
                        module.q_proj_weight,
                        module.k_proj_weight,
                        module.v_proj_weight,
                    )
                    if weight is not None
                ]
            )
            if not self._apply_override("attention.qkv", "weight", qkv_weights):
                if module.in_proj_weight is not None:
                    self._xavier_uniform_(module.in_proj_weight)
                else:
                    self._joint_xavier_uniform_(qkv_weights)

            attention_biases = [module.in_proj_bias, module.bias_k, module.bias_v]
            if not self._apply_override("attention.qkv", "bias", attention_biases):
                self._zero_(module.in_proj_bias)
                if module.bias_k is not None and not self._is_initialized(
                    module.bias_k
                ):
                    nn.init.xavier_normal_(module.bias_k)
                    self._mark_initialized(module.bias_k)
                if module.bias_v is not None and not self._is_initialized(
                    module.bias_v
                ):
                    nn.init.xavier_normal_(module.bias_v)
                    self._mark_initialized(module.bias_v)

            self._initialize_linear_xavier(module.out_proj, "attention.output")

    def _initialize_remaining_modules(self) -> None:
        for module in self.model.modules():
            if isinstance(module, nn.Embedding):
                group: LayerGroup = (
                    "embedding.position"
                    if id(module) in self.position_embedding_module_ids
                    else "embedding.input"
                )
                if not self._apply_override(group, "weight", [module.weight]):
                    self._normal_(module.weight, EMBEDDING_INIT_STD)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
            elif isinstance(module, (nn.LayerNorm, RMSNorm)):
                weight = getattr(module, "weight", None)
                if not self._apply_override("normalization", "weight", [weight]):
                    if isinstance(weight, Tensor) and not self._is_initialized(weight):
                        nn.init.ones_(weight)
                        self._mark_initialized(weight)
                bias = getattr(module, "bias", None)
                if not self._apply_override("normalization", "bias", [bias]):
                    self._zero_(bias)
            elif isinstance(module, nn.Linear):
                group = (
                    "ingestion.output_projection"
                    if id(module) in self.ingestion_projection_module_ids
                    else "fallback.linear"
                )
                self._initialize_linear_xavier(module, group)
            elif isinstance(module, CONVOLUTION_TYPES):
                self._initialize_conv_xavier(module, "fallback.convolution")

    def _initialize_free_parameters(self) -> None:
        for _, parameter in self.model.named_parameters(remove_duplicate=True):
            if self._is_initialized(parameter):
                continue
            if not parameter.is_floating_point():
                raise TypeError(
                    "Sequifier only supports floating-point trainable parameters, "
                    f"got {parameter.dtype}."
                )
            if not self._apply_override("free_parameter", "weight", [parameter]):
                self._normal_(parameter, EMBEDDING_INIT_STD)


def initialize_model_weights(
    model: nn.Module,
    initialization: Optional[ModelInitializationConfig] = None,
) -> None:
    """Initialize model parameters using current defaults plus configured overrides."""

    _ModelWeightInitializer(model, initialization).initialize()
