"""Capture and restore Python, NumPy, Torch, and CUDA randomness."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sequifier.artifacts.run_checkpoint import DistributedRandomState


@dataclass(frozen=True)
class RandomState:
    python: Any
    numpy: Any
    torch_cpu: torch.Tensor
    cuda: tuple[torch.Tensor, ...] | None
    mps: torch.Tensor | None = None
    all_devices: bool = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "numpy": self.numpy,
            "torch_cpu": self.torch_cpu,
            "cuda": self.cuda,
            "mps": self.mps,
            "all_devices": self.all_devices,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "RandomState":
        return cls(**state)


class RandomStateManager:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def capture_local(self, *, all_devices: bool = False) -> RandomState:
        return RandomState(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch_cpu=torch.get_rng_state(),
            cuda=(
                tuple(torch.cuda.get_rng_state_all())
                if all_devices and torch.cuda.is_available()
                else (torch.cuda.get_rng_state(self.device),)
                if self.device.type == "cuda" and torch.cuda.is_available()
                else None
            ),
            mps=(
                torch.mps.get_rng_state()
                if (all_devices or self.device.type == "mps")
                and torch.backends.mps.is_available()
                else None
            ),
            all_devices=all_devices,
        )

    def gather(self, distributed: Any) -> DistributedRandomState:
        local = self.capture_local().state_dict()
        return DistributedRandomState(states=tuple(distributed.gather_objects(local)))

    def select_for_rank(self, state: DistributedRandomState, rank: int) -> RandomState:
        if not state.states:
            raise ValueError("Checkpoint contains no random states.")
        if rank >= len(state.states):
            raise ValueError(
                f"Checkpoint has {len(state.states)} random states for rank {rank}."
            )
        return RandomState.from_state_dict(state.states[rank])

    def restore(self, state: RandomState) -> None:
        if state.all_devices:
            if state.cuda is not None and (
                not torch.cuda.is_available()
                or len(state.cuda) != torch.cuda.device_count()
            ):
                raise ValueError("checkpoint CUDA RNG topology differs")
            if state.mps is not None and not torch.backends.mps.is_available():
                raise ValueError("checkpoint requires the saved MPS RNG")
        random.setstate(state.python)
        np.random.set_state(state.numpy)
        torch.set_rng_state(state.torch_cpu)
        if state.all_devices and state.cuda is not None:
            torch.cuda.set_rng_state_all(list(state.cuda))
        elif (
            state.cuda is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            device_index = self.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            cuda_state = (
                state.cuda[0] if len(state.cuda) == 1 else state.cuda[device_index]
            )
            torch.cuda.set_rng_state(cuda_state, self.device)
        if (
            state.mps is not None
            and (state.all_devices or self.device.type == "mps")
            and torch.backends.mps.is_available()
        ):
            torch.mps.set_rng_state(state.mps)


def capture_rng_state(device):
    return RandomStateManager(device).capture_local(all_devices=True).state_dict()


def restore_rng_state(device, state):
    RandomStateManager(device).restore(RandomState.from_state_dict(state))
