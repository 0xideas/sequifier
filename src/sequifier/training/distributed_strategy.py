"""Distributed execution and state-dict strategies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, TypeVar, runtime_checkable

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from sequifier.artifacts.state_dict import canonicalize_state_dict


@dataclass(frozen=True)
class PreparedNetwork:
    network: nn.Module
    callable_network: nn.Module


@runtime_checkable
class DistributedStrategy(Protocol):
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    def broadcast(self, value: Any, source: int = 0) -> Any: ...
    def sum(self, value: float) -> float: ...
    def clip_gradients(self, network: nn.Module, limit: float) -> Tensor: ...
    def prepare_frozen(self, network: nn.Module) -> PreparedNetwork: ...
    def prepare_network(self, network: nn.Module) -> PreparedNetwork: ...
    def prepare_optimizer_parameters(
        self, network: nn.Module
    ) -> Iterable[nn.Parameter]: ...
    def capture_model_state(self, network: nn.Module) -> dict[str, Tensor]: ...
    def capture_optimizer_state(
        self, network: nn.Module, optimizer: Optimizer
    ) -> dict[str, Any]: ...
    def restore_model_state(
        self, network: nn.Module, state: dict[str, Tensor]
    ) -> None: ...
    def restore_optimizer_state(
        self, network: nn.Module, optimizer: Optimizer, state: dict[str, Any]
    ) -> None: ...
    def gather_objects(self, value: Any) -> list[Any]: ...
    def gather_for_writer(self, value: Any) -> list[Any]: ...
    def barrier(self) -> None: ...
    def finalize(self) -> None: ...


WriterResult = TypeVar("WriterResult")


def on_writer(
    strategy: DistributedStrategy, operation: Callable[[], WriterResult]
) -> WriterResult:
    """Publish on rank zero and acknowledge success before workers advance."""
    if strategy.world_size == 1:
        return operation()
    message = None
    if strategy.rank == 0:
        try:
            message = True, operation()
        except Exception as error:
            strategy.broadcast((False, str(error)))
            raise
    success, result = strategy.broadcast(message)
    if not success:
        raise RuntimeError(f"Rank zero failed: {result}")
    return result


@dataclass
class LocalStrategy:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")

    def broadcast(self, value: Any, source: int = 0) -> Any:
        return value

    def sum(self, value):
        return value

    def clip_gradients(self, network, limit):
        return nn.utils.clip_grad_norm_(network.parameters(), limit)

    def prepare_frozen(self, network):
        return PreparedNetwork(network, network)

    def prepare_network(self, network: nn.Module) -> PreparedNetwork:
        return PreparedNetwork(network, network)

    def prepare_optimizer_parameters(
        self, network: nn.Module
    ) -> Iterable[nn.Parameter]:
        return network.parameters()

    def capture_model_state(self, network: nn.Module) -> dict[str, Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in canonicalize_state_dict(network.state_dict()).items()
        }

    def capture_optimizer_state(
        self, network: nn.Module, optimizer: Optimizer
    ) -> dict[str, Any]:
        return optimizer.state_dict()

    def restore_model_state(self, network: nn.Module, state: dict[str, Tensor]) -> None:
        network.load_state_dict(canonicalize_state_dict(state))

    def restore_optimizer_state(
        self, network: nn.Module, optimizer: Optimizer, state: dict[str, Any]
    ) -> None:
        optimizer.load_state_dict(state)
        for name, parameter in network.named_parameters():
            if ".prediction_scales.log_scales." in name:
                slots = optimizer.state.get(parameter, {})
                for key, value in slots.items():
                    if key != "step" and isinstance(value, Tensor) and value.ndim == 0:
                        slots[key] = value.reshape(1)

    def gather_objects(self, value: Any) -> list[Any]:
        return [value]

    def gather_for_writer(self, value):
        return [value]

    def barrier(self) -> None:
        return None

    def finalize(self) -> None:
        return None


@dataclass
class DistributedDataParallelStrategy(LocalStrategy):
    find_unused_parameters: bool = False
    control_group: Any = None

    def broadcast(self, value: Any, source: int = 0) -> Any:
        payload = [value if self.rank == source else None]
        dist.broadcast_object_list(payload, src=source, group=self.control_group)
        return payload[0]

    def sum(self, value):
        total = torch.tensor(value, dtype=torch.float64, device=self.device)
        dist.all_reduce(total)
        return total.item()

    def prepare_frozen(self, network):
        with torch.no_grad():
            for value in (*network.parameters(), *network.buffers()):
                dist.broadcast(value, src=0)
        return PreparedNetwork(network, network)

    def prepare_network(self, network: nn.Module) -> PreparedNetwork:
        device_ids = [self.local_rank] if self.device.type == "cuda" else None
        wrapped = DistributedDataParallel(
            network,
            device_ids=device_ids,
            find_unused_parameters=self.find_unused_parameters,
        )
        return PreparedNetwork(network, wrapped)

    def gather_objects(self, value: Any) -> list[Any]:
        gathered: list[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, value, group=self.control_group)
        return gathered

    def gather_for_writer(self, value):
        gathered = [None] * self.world_size if self.rank == 0 else None
        dist.gather_object(value, gathered, dst=0, group=self.control_group)
        return gathered or []

    def barrier(self) -> None:
        dist.barrier()

    def finalize(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


@dataclass
class FullyShardedStrategy(DistributedDataParallelStrategy):
    cpu_offload: bool = False
    mixed_precision_dtype: torch.dtype | None = None
    preserve_io_dtype: bool = False

    def prepare_network(self, network: nn.Module) -> PreparedNetwork:
        from packaging import version

        if version.parse(torch.__version__) >= version.parse("2.6.0"):
            from torch.distributed.fsdp import (
                CPUOffloadPolicy,
                MixedPrecisionPolicy,
                fully_shard,
            )
        else:
            from torch.distributed._composable.fsdp import (  # type: ignore
                CPUOffloadPolicy,
                MixedPrecisionPolicy,
                fully_shard,
            )
        from torch.distributed.device_mesh import DeviceMesh

        group = dist.group.WORLD
        assert group is not None
        mesh = DeviceMesh.from_group(group, self.device.type)
        kwargs: dict[str, Any] = {"mesh": mesh, "reshard_after_forward": True}
        # FSDP2 shards the initial tensors independently on each rank. Initialize
        # each shard from the same weights, including randomly initialized critics.
        with torch.no_grad():
            for value in (*network.parameters(), *network.buffers()):
                initial = value.detach().to(self.device).contiguous()
                dist.broadcast(initial, src=0)
                if self.rank != 0:
                    value.copy_(initial)
                del initial

        if self.mixed_precision_dtype is not None:
            kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=self.mixed_precision_dtype,
                reduce_dtype=self.mixed_precision_dtype,
                output_dtype=self.mixed_precision_dtype,
            )
        else:
            kwargs["mp_policy"] = MixedPrecisionPolicy()
        if self.cpu_offload:
            kwargs["offload_policy"] = CPUOffloadPolicy()
        from sequifier.model.network import ComposableTransformerNetwork

        # A policy can decode a different number of prefixes while scoring or
        # selecting. Only the fixed history backbone has nested collectives.
        layers = dict.fromkeys(
            layer
            for module in network.modules()
            if isinstance(module, ComposableTransformerNetwork)
            for layer in module.backbone.layers
        )
        for layer in layers:
            fully_shard(layer, **kwargs)
        if self.mixed_precision_dtype is not None and self.preserve_io_dtype:
            # Policy inputs include recorded actions in domain units. Scoring
            # outputs may also require more precision than the model parameters.
            kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=self.mixed_precision_dtype,
                reduce_dtype=self.mixed_precision_dtype,
                cast_forward_inputs=False,
            )
        fully_shard(network, **kwargs)
        self.barrier()
        return PreparedNetwork(network, network)

    @torch.no_grad()
    def clip_gradients(self, network, limit):
        # Every FSDP2 parameter in this strategy is sharded across the world mesh.
        gradients = [
            p.grad.to_local() for p in network.parameters() if p.grad is not None
        ]
        squares = [
            torch.linalg.vector_norm(g, dtype=torch.float64).square() for g in gradients
        ]
        norm = self.sum(float(torch.stack(squares).sum()) if squares else 0.0) ** 0.5
        if norm > limit:
            for gradient in gradients:
                gradient.mul_(limit / (norm + 1e-6))
        return torch.tensor(norm, device=self.device, dtype=torch.float64)

    def prepare_frozen(self, network):
        return self.prepare_network(network)

    def capture_model_state(self, network: nn.Module) -> dict[str, Tensor]:
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        return canonicalize_state_dict(get_model_state_dict(network, options=options))

    def capture_optimizer_state(
        self, network: nn.Module, optimizer: Optimizer
    ) -> dict[str, Any]:
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        # DCP initializes an empty optimizer by stepping it with zero gradients.
        # An untouched optimizer must retain its lazy state across a checkpoint.
        if not optimizer.state:
            return optimizer.state_dict()
        return get_optimizer_state_dict(network, optimizer, options=options)

    def restore_model_state(self, network: nn.Module, state: dict[str, Tensor]) -> None:
        state = canonicalize_state_dict(state)
        for key, tensor in state.items():
            if ".prediction_scales.log_scales." in key and tensor.ndim == 0:
                state[key] = tensor.reshape(1)
        if self.cpu_offload:
            from torch.distributed.tensor import DTensor, distribute_tensor

            local = network.state_dict()
            for key, tensor in state.items():
                target = local.get(key)
                if isinstance(target, DTensor):
                    shard = distribute_tensor(
                        tensor, target.device_mesh, target.placements
                    )
                    state[key] = shard.to(target.device)
            network.load_state_dict(state)
            return
        set_model_state_dict(
            network,
            dict(state),
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

    def restore_optimizer_state(
        self, network: nn.Module, optimizer: Optimizer, state: dict[str, Any]
    ) -> None:
        state = dict(state)
        slots = dict(state["state"])
        if not slots:
            # SGD without momentum has no tensor state for DCP to distribute.
            optimizer.load_state_dict(state)
            return
        for name, values in slots.items():
            if isinstance(name, str) and ".prediction_scales.log_scales." in name:
                slots[name] = {
                    key: value.reshape(1)
                    if key != "step" and isinstance(value, Tensor) and value.ndim == 0
                    else value
                    for key, value in values.items()
                }
        state["state"] = slots
        # DCP allocates destination slots for every trainable parameter. A route
        # that never received a gradient must still have no momentum or step.
        inactive = [
            parameter
            for group, saved_group in zip(
                optimizer.param_groups, state["param_groups"], strict=True
            )
            for parameter, name in zip(
                group["params"], saved_group["params"], strict=True
            )
            if name not in slots and parameter.requires_grad
        ]
        for parameter in inactive:
            parameter.requires_grad_(False)
        try:
            set_optimizer_state_dict(
                network,
                optimizer,
                optim_state_dict=state,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
        finally:
            for parameter in inactive:
                parameter.requires_grad_(True)


def strategy_for(
    environment,
    settings,
    *,
    find_unused=False,
    mixed_precision_dtype=None,
    preserve_io_dtype=False,
) -> DistributedStrategy:
    if not environment.distributed:
        return LocalStrategy(device=environment.device)
    options = dict(
        rank=environment.rank,
        local_rank=environment.local_rank,
        world_size=environment.world_size,
        device=environment.device,
        control_group=environment.control_group,
    )
    if settings.strategy == "fsdp":
        return FullyShardedStrategy(
            **options,
            cpu_offload=settings.cpu_offload,
            mixed_precision_dtype=mixed_precision_dtype,
            preserve_io_dtype=preserve_io_dtype,
        )
    return DistributedDataParallelStrategy(
        **options, find_unused_parameters=find_unused
    )
