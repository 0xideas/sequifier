"""Distributed execution and state-dict strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, runtime_checkable

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
    def barrier(self) -> None: ...
    def finalize(self) -> None: ...


@dataclass
class LocalStrategy:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

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

    def gather_objects(self, value: Any) -> list[Any]:
        return [value]

    def barrier(self) -> None:
        return None

    def finalize(self) -> None:
        return None


@dataclass
class DistributedDataParallelStrategy(LocalStrategy):
    device: torch.device = torch.device("cpu")
    find_unused_parameters: bool = False

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
        dist.all_gather_object(gathered, value)
        return gathered

    def barrier(self) -> None:
        dist.barrier()

    def finalize(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


@dataclass
class FullyShardedStrategy(DistributedDataParallelStrategy):
    cpu_offload: bool = False
    mixed_precision_dtype: torch.dtype | None = None

    def prepare_network(self, network: nn.Module) -> PreparedNetwork:
        from packaging import version

        if version.parse(torch.__version__) >= version.parse("2.6.0"):
            from torch.distributed.fsdp import (
                MixedPrecisionPolicy,
                OffloadPolicy,
                fully_shard,
            )
        else:
            from torch.distributed._composable.fsdp import (  # type: ignore
                MixedPrecisionPolicy,
                OffloadPolicy,
                fully_shard,
            )
        kwargs: dict[str, Any] = {}
        if self.mixed_precision_dtype is not None:
            kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=self.mixed_precision_dtype,
                reduce_dtype=self.mixed_precision_dtype,
                output_dtype=self.mixed_precision_dtype,
            )
        else:
            kwargs["mp_policy"] = MixedPrecisionPolicy()
        if self.cpu_offload:
            kwargs["offload_policy"] = OffloadPolicy()
        seen: set[int] = set()
        backbone = getattr(network, "backbone", None)
        for layer in getattr(backbone, "layers", ()):
            if id(layer) not in seen:
                fully_shard(layer, **kwargs)
                seen.add(id(layer))
        fully_shard(network, **kwargs)
        self.barrier()
        return PreparedNetwork(network, network)

    def capture_model_state(self, network: nn.Module) -> dict[str, Tensor]:
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        return canonicalize_state_dict(get_model_state_dict(network, options=options))

    def capture_optimizer_state(
        self, network: nn.Module, optimizer: Optimizer
    ) -> dict[str, Any]:
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        return get_optimizer_state_dict(network, optimizer, options=options)

    def restore_model_state(self, network: nn.Module, state: dict[str, Tensor]) -> None:
        set_model_state_dict(
            network,
            canonicalize_state_dict(state),
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

    def restore_optimizer_state(
        self, network: nn.Module, optimizer: Optimizer, state: dict[str, Any]
    ) -> None:
        set_optimizer_state_dict(
            network,
            optimizer,
            optim_state_dict=state,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
