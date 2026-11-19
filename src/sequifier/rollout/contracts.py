"""Adapter values are domain labels or finite scalars in original units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from sequifier.contract_versions import FORECAST_ROLLOUT_PROTOCOL, ContractVersions
from sequifier.io.values import DomainValue


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


class ForecastRolloutAdapterBase:
    """Supply the protocol metadata for a numeric forecast adapter."""

    protocol = FORECAST_ROLLOUT_PROTOCOL
    protocol_version = ContractVersions.FORECAST_ROLLOUT_ADAPTER


class EnvironmentAdapter(Protocol):
    protocol: str
    protocol_version: int

    def reset(self, request: ResetRequest) -> None:
        """Initialize from the supplied history; reset must not replace the seed."""
        ...

    def step(self, request: StepRequest) -> StepResult: ...

    def close(self) -> None: ...
