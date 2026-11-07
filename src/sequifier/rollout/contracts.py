"""Adapter values are domain labels or finite scalars in original units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Protocol

from sequifier.io.values import DomainValue

if TYPE_CHECKING:
    from sequifier.trajectory.schema import Transition


@dataclass(frozen=True)
class ResetRequest:
    episode_id: str
    seed_history: Mapping[str, tuple[DomainValue, ...]]
    attention_valid_mask: tuple[bool, ...]
    static_inputs: Mapping[str, DomainValue]
    seed: int


@dataclass(frozen=True)
class StepRequest:
    episode_id: str
    step_id: int
    observation: Mapping[str, DomainValue]
    predictions: Mapping[str, DomainValue]
    static_inputs: Mapping[str, DomainValue]


@dataclass(frozen=True)
class StepResult:
    external_inputs: Mapping[str, DomainValue]
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
