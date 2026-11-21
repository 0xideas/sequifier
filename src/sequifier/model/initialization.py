"""Centralized, configurable trainable-parameter initialization."""

import math
import warnings
from collections.abc import Iterable, Sequence
from typing import Optional

import torch
from torch import Tensor, nn

from sequifier.config.initialization_config import (
    IdentityPlusNormalInitialization,
    InitializationMethodConfig,
    InitializationTarget,
    LayerGroup,
    ModelInitializationConfig,
    NormalInitialization,
    OnesInitialization,
    ParameterKind,
    XavierUniformInitialization,
    ZerosInitialization,
)
from sequifier.model.ingestions import BaseFeatureIngestion
from sequifier.model.parameter_groups import semantic_batches
from sequifier.typechecking import beartype

_DEFAULT_METHODS: dict[str, InitializationMethodConfig] = {
    "normal": NormalInitialization(method="normal", std=0.02),
    "ones": OnesInitialization(method="ones"),
    "zeros": ZerosInitialization(method="zeros"),
    "identity_plus_normal": IdentityPlusNormalInitialization(
        method="identity_plus_normal", std=0.02
    ),
    "xavier_uniform": XavierUniformInitialization(method="xavier_uniform", gain=1.0),
}


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

    def initialize(self) -> None:
        for batch in semantic_batches(self.model):
            pending = [
                p
                for p in dict.fromkeys(batch.parameters)
                if p is not None and id(p) not in self.initialized_parameter_ids
            ]
            if pending:
                override = self.initialization.override_for(batch.group)
                method = getattr(override, batch.kind, None)
                if method is not None:
                    self._apply_method(batch.group, batch.kind, pending, method)
                    self.matched_override_targets.add((batch.group, batch.kind))
                else:
                    self._apply_default(batch, pending)
                self.initialized_parameter_ids.update(map(id, pending))
            if batch.padding_idx is not None:
                weight = batch.parameters[0]
                assert weight is not None
                with torch.no_grad():
                    weight[batch.padding_idx].zero_()

    def _apply_default(self, batch, pending):
        if batch.default == "joint_xavier":
            fan_out = sum(_fan_out(p) for p in batch.parameters if p is not None)
            self._joint_xavier_uniform_raw(pending, fan_out=fan_out)
            return
        if batch.default == "attention_bias":
            for parameter in pending:
                if parameter is batch.parameters[0]:
                    nn.init.zeros_(parameter)
                else:
                    nn.init.xavier_normal_(parameter)
            return
        self._apply_method(
            batch.group, batch.kind, pending, _DEFAULT_METHODS[batch.default]
        )

    @beartype
    def _apply_method(
        self,
        group: LayerGroup,
        parameter_kind: ParameterKind,
        tensors: Sequence[Tensor],
        config: InitializationMethodConfig,
    ) -> None:
        if config.method == "preserve":
            return

        for parameter in tensors:
            if not parameter.is_floating_point():
                raise TypeError(
                    f"Initialization for {group}.{parameter_kind} requires a "
                    f"floating-point tensor, got {parameter.dtype}."
                )

        try:
            if config.method == "normal":
                for parameter in tensors:
                    nn.init.normal_(parameter, mean=config.mean, std=config.std)
            elif config.method == "uniform":
                for parameter in tensors:
                    nn.init.uniform_(parameter, a=config.low, b=config.high)
            elif config.method == "xavier_uniform":
                if config.fan_mode == "joint":
                    self._joint_xavier_uniform_raw(tensors, gain=config.gain)
                else:
                    for parameter in tensors:
                        nn.init.xavier_uniform_(parameter, gain=config.gain)
            elif config.method == "xavier_normal":
                if config.fan_mode == "joint":
                    self._joint_xavier_normal_raw(tensors, gain=config.gain)
                else:
                    for parameter in tensors:
                        nn.init.xavier_normal_(parameter, gain=config.gain)
            elif config.method == "kaiming_uniform":
                for parameter in tensors:
                    nn.init.kaiming_uniform_(
                        parameter,
                        a=config.a,
                        mode=config.mode,
                        nonlinearity=config.nonlinearity,
                    )
            elif config.method == "kaiming_normal":
                for parameter in tensors:
                    nn.init.kaiming_normal_(
                        parameter,
                        a=config.a,
                        mode=config.mode,
                        nonlinearity=config.nonlinearity,
                    )
            elif config.method == "constant":
                for parameter in tensors:
                    nn.init.constant_(parameter, config.value)
            elif config.method == "zeros":
                for parameter in tensors:
                    nn.init.zeros_(parameter)
            elif config.method == "ones":
                for parameter in tensors:
                    nn.init.ones_(parameter)
            elif config.method == "identity_plus_normal":
                for parameter in tensors:
                    self._identity_plus_normal_raw(
                        parameter,
                        mean=config.mean,
                        std=config.std,
                    )
            else:
                raise ValueError(f"Unknown initialization method: {config.method}")
        except (RuntimeError, ValueError) as error:
            shapes = [tuple(parameter.shape) for parameter in tensors]
            raise ValueError(
                f"Cannot apply {config.method!r} to {group}.{parameter_kind} "
                f"with tensor shapes {shapes}: {error}"
            ) from error

    @staticmethod
    @beartype
    def _joint_xavier_uniform_raw(
        parameters: Iterable[Tensor],
        *,
        gain: float = 1.0,
        fan_out: int | None = None,
    ) -> None:
        weights = list(parameters)
        joint_fan_out = (
            sum(_fan_out(weight) for weight in weights) if fan_out is None else fan_out
        )
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
            targets = {
                (batch.group, batch.kind)
                for batch in semantic_batches(node)
                if any(p is not None for p in batch.parameters)
            }
            unmatched = config.initialization.configured_targets() - targets
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
