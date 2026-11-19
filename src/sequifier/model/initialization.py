"""Centralized, configurable trainable-parameter initialization."""

import math
import warnings
from collections.abc import Iterable
from typing import Optional, TypeAlias, cast

import torch
from torch import Tensor, nn

from sequifier.config.initialization_config import (
    InitializationMethodConfig,
    InitializationTarget,
    LayerGroup,
    ModelInitializationConfig,
    ParameterKind,
)
from sequifier.model.ingestions import (
    BaseFeatureIngestion,
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

EMBEDDING_INIT_STD = 0.02
CONVOLUTION_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
Convolution: TypeAlias = nn.Conv1d | nn.Conv2d | nn.Conv3d


@beartype
def _fan_in(weight: Tensor) -> int:
    if weight.ndim < 2:
        raise ValueError(
            f"Fan-in initialization requires at least two dimensions, got {weight.shape}."
        )
    return weight.shape[1] * math.prod(weight.shape[2:])


@beartype
def _fan_out(weight: Tensor) -> int:
    if weight.ndim < 2:
        raise ValueError(
            f"Fan-out initialization requires at least two dimensions, got {weight.shape}."
        )
    return weight.shape[0] * math.prod(weight.shape[2:])


class _ModelWeightInitializer:
    """Apply current defaults with optional semantic layer-group overrides."""

    @beartype
    def __init__(
        self,
        model: nn.Module,
        initialization: Optional[ModelInitializationConfig],
    ):
        self.model = model
        self.initialization = initialization or ModelInitializationConfig()
        self.initialized_parameter_ids: set[int] = set()
        self.matched_override_targets: set[InitializationTarget] = set()
        self.position_embedding_module_ids = position_embedding_ids(model)
        self.ingestion_projection_module_ids = ingestion_projection_ids(model)
        self.decoder_output_module_ids = decoder_output_ids(decoder_module(model))

    @beartype
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

    @beartype
    def _is_initialized(self, parameter: Optional[Tensor]) -> bool:
        return parameter is None or id(parameter) in self.initialized_parameter_ids

    @beartype
    def _mark_initialized(self, parameter: Optional[Tensor]) -> None:
        if parameter is not None:
            self.initialized_parameter_ids.add(id(parameter))

    @beartype
    def _override_method(
        self,
        group: LayerGroup,
        parameter_kind: ParameterKind,
    ) -> Optional[InitializationMethodConfig]:
        override = self.initialization.override_for(group)
        if override is None:
            return None
        if parameter_kind == "weight":
            return override.weight
        if parameter_kind == "bias":
            return override.bias
        raise ValueError(f"Unknown parameter kind: {parameter_kind}")

    @beartype
    def _apply_override(
        self,
        group: LayerGroup,
        parameter_kind: ParameterKind,
        parameters: Iterable[Optional[Tensor]],
    ) -> bool:
        method = self._override_method(group, parameter_kind)
        if method is None:
            return False
        self._apply_method(group, parameter_kind, parameters, method)
        return True

    @beartype
    def _apply_method(
        self,
        group: LayerGroup,
        parameter_kind: ParameterKind,
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

        self.matched_override_targets.add((group, parameter_kind))

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
            elif method == "xavier_uniform":
                if config.fan_mode == "joint":
                    self._joint_xavier_uniform_raw(tensors, gain=config.gain)
                else:
                    for parameter in tensors:
                        nn.init.xavier_uniform_(parameter, gain=config.gain)
            elif method == "xavier_normal":
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
    @beartype
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
    @beartype
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
    @beartype
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

    @beartype
    def _zero_(self, parameter: Optional[Tensor]) -> None:
        if parameter is None or self._is_initialized(parameter):
            return
        nn.init.zeros_(parameter)
        self._mark_initialized(parameter)

    @beartype
    def _normal_(self, parameter: Tensor, std: float) -> None:
        if self._is_initialized(parameter):
            return
        nn.init.normal_(parameter, mean=0.0, std=std)
        self._mark_initialized(parameter)

    @beartype
    def _xavier_uniform_(self, parameter: Tensor, gain: float = 1.0) -> None:
        if self._is_initialized(parameter):
            return
        nn.init.xavier_uniform_(parameter, gain=gain)
        self._mark_initialized(parameter)

    @beartype
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

    @beartype
    def _initialize_linear_xavier(
        self,
        layer: nn.Linear,
        group: LayerGroup,
    ) -> None:
        if not self._apply_override(group, "weight", [layer.weight]):
            self._xavier_uniform_(layer.weight)
        if not self._apply_override(group, "bias", [layer.bias]):
            self._zero_(layer.bias)

    @beartype
    def _initialize_conv_xavier(
        self,
        layer: Convolution,
        group: LayerGroup,
    ) -> None:
        if not self._apply_override(group, "weight", [layer.weight]):
            self._xavier_uniform_(layer.weight)
        if not self._apply_override(group, "bias", [layer.bias]):
            self._zero_(layer.bias)

    @beartype
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

    @beartype
    def _initialize_feed_forward(self, feed_forward: FeedForward) -> None:
        if feed_forward.activation == "swiglu":
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

    @beartype
    def _initialize_decoder(self) -> None:
        decoder = decoder_module(self.model)
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

    @beartype
    def _initialize_range_position_projection(self) -> None:
        projection = range_position_projection(self.model)
        if projection is None:
            return
        if not isinstance(projection, nn.Linear):
            raise TypeError("range_position_projection must be an nn.Linear")
        linear_projection = cast(nn.Linear, projection)
        if linear_projection.out_features + 1 != linear_projection.in_features:
            raise ValueError(
                "range_position_projection must map model_dim + 1 to model_dim"
            )

        if not self._apply_override(
            "position.range_projection", "weight", [linear_projection.weight]
        ):
            if not self._is_initialized(linear_projection.weight):
                self._identity_plus_normal_raw(
                    linear_projection.weight,
                    mean=0.0,
                    std=EMBEDDING_INIT_STD,
                )
                self._mark_initialized(linear_projection.weight)
        if not self._apply_override(
            "position.range_projection", "bias", [linear_projection.bias]
        ):
            self._zero_(linear_projection.bias)

    @beartype
    def _initialize_real_feature_projections(self) -> None:
        for module in self.model.modules():
            if isinstance(module, RealFeatureProjection):
                self._initialize_linear_xavier(module, "real_feature_projection")

    @beartype
    def _initialize_temporal_convolutions(self) -> None:
        for module in self.model.modules():
            if not isinstance(module, TemporalConvFeatureIngestion):
                continue
            for layer in module.layers:
                if isinstance(layer, CONVOLUTION_TYPES):
                    self._initialize_conv_xavier(layer, "temporal_convolution")

    @beartype
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

    @beartype
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
                weight = module.weight
                if not self._apply_override("normalization", "weight", [weight]):
                    if isinstance(weight, Tensor) and not self._is_initialized(weight):
                        nn.init.ones_(weight)
                        self._mark_initialized(weight)
                bias = module.bias
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

    @beartype
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


@beartype
def initialize_model_weights(
    model: nn.Module,
    initialization: Optional[ModelInitializationConfig] = None,
    *,
    warn_unmatched: bool = True,
) -> set[InitializationTarget]:
    """Initialize parameters and return the configured targets that matched."""

    initializer = _ModelWeightInitializer(model, initialization)
    initializer.initialized_parameter_ids.update(
        id(parameter) for parameter in model.parameters() if not parameter.requires_grad
    )
    initializer.initialize()
    unmatched = initializer.initialization.configured_targets().difference(
        initializer.matched_override_targets
    )
    if warn_unmatched and unmatched:
        targets = ", ".join(f"{group}.{kind}" for group, kind in sorted(unmatched))
        warnings.warn(
            f"Initialization overrides matched no parameters: {targets}",
            stacklevel=2,
        )
    return initializer.matched_override_targets


def initialize_owned_ingestion(
    model: BaseFeatureIngestion, initialization, *, warn_unmatched=True
):
    from sequifier.model.initialization_seed import isolated_initialization

    def active(node):
        if node.initialization_seed is not None:
            return True
        for child in node.initialization_children():
            child_config = child.initialization_config
            if (
                child_config is not None
                and child_config.initialization.configured_targets()
            ) or active(child):
                return True
        return False

    if not active(model):
        return initialize_model_weights(
            model, initialization, warn_unmatched=warn_unmatched
        )
    all_matched = set()
    visited = set()
    requests = {}

    def visit(node, inherited):
        config = node.initialization_config
        values = inherited.model_dump(mode="python", exclude_none=True)
        if config is not None:
            for group, targets in config.initialization.model_dump(
                mode="python", exclude_none=True
            ).items():
                values[group] = {**values.get(group, {}), **targets}
        effective = ModelInitializationConfig.model_validate(values)
        branches = node.initialization_children()
        excluded = {id(p) for child in branches for p in child.parameters()}
        owned = {id(p) for p in node.parameters()} - excluded
        request = (
            effective.model_dump(mode="python"),
            node.initialization_seed,
        )
        for identity in owned:
            if identity in requests and requests[identity] != request:
                raise ValueError(
                    "Conflicting initialization policy or seed for a shared parameter alias"
                )
            requests[identity] = request
        if config is not None:
            from sequifier.model.parameter_groups import semantic_parameter_groups

            groups = semantic_parameter_groups(node)
            biases = {
                id(parameter)
                for module in node.modules()
                for name, parameter in module.named_parameters(
                    recurse=False, remove_duplicate=False
                )
                if name == "bias"
            }
            unmatched = {
                (group, kind)
                for group, kind in config.initialization.configured_targets()
                if not any(
                    (id(p) in biases) == (kind == "bias") for p in groups.get(group, ())
                )
            }
            if unmatched:
                warnings.warn(
                    f"Initialization overrides matched no parameters in ingestion scope {config.type!r}: {sorted(unmatched)}",
                    stacklevel=2,
                )
        initializer = _ModelWeightInitializer(node, effective)
        initializer.initialized_parameter_ids = (
            visited
            | excluded
            | {
                id(parameter)
                for parameter in node.parameters()
                if not parameter.requires_grad
            }
        )
        with isolated_initialization(node.initialization_seed):
            initializer.initialize()
        visited.update(id(p) for p in node.parameters() if id(p) not in excluded)
        all_matched.update(initializer.matched_override_targets)
        for child in branches:
            visit(child, effective)

    visit(model, initialization)
    return all_matched
