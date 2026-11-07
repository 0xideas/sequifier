"""Adapter values are finite numeric scalars in original (unscaled) units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Protocol

if TYPE_CHECKING:
    from sequifier.trajectory.schema import Transition


@dataclass(frozen=True)
class ResetRequest:
    episode_id: str
    seed_history: Mapping[str, tuple[float, ...]]
    attention_valid_mask: tuple[bool, ...]
    static_inputs: Mapping[str, float]
    seed: int


@dataclass(frozen=True)
class StepRequest:
    episode_id: str
    step_id: int
    observation: Mapping[str, float]
    predictions: Mapping[str, float]
    static_inputs: Mapping[str, float]


@dataclass(frozen=True)
class StepResult:
    external_inputs: Mapping[str, float]
    reward: float
    terminated: bool = False
    truncated: bool = False


class EnvironmentAdapter(Protocol):
    protocol_version: int

    def reset(self, request: ResetRequest) -> None:
        """Initialize from the supplied history; reset must not replace the seed."""
        ...

    def step(self, request: StepRequest) -> StepResult: ...

    def close(self) -> None: ...


class RolloutObserver(Protocol):
    """Per-episode state updated only after an executed, recorded transition."""

    def reset(self, request: ResetRequest) -> None: ...

    def observe(self, transition: Transition) -> None: ...
