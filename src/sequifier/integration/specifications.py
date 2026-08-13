from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RankPolicy = Literal["all", "rank_zero", "all_reduce_summary"]
StatePolicy = Literal["required", "if_present", "fresh"]


@dataclass(frozen=True)
class ExecutionRequirements:
    activation_tracing: bool = False
    interventions: bool = False
    higher_order_gradients: bool = False
    full_parameters: bool = False


@dataclass(frozen=True)
class IntegrationSpec:
    integration_id: str
    factory: str
    config: dict[str, Any] = field(default_factory=dict)
    rank_policy: RankPolicy = "all"
    state_policy: StatePolicy = "if_present"

    def __post_init__(self) -> None:
        if not self.integration_id.strip():
            raise ValueError("integration_id must not be empty.")
        module_name, separator, attribute_name = self.factory.partition(":")
        if not separator or not module_name or not attribute_name:
            raise ValueError(
                "factory must be an explicit 'module:attribute' import path."
            )
        if self.rank_policy not in {"all", "rank_zero", "all_reduce_summary"}:
            raise ValueError(f"Unsupported rank_policy: {self.rank_policy!r}.")
        if self.state_policy not in {"required", "if_present", "fresh"}:
            raise ValueError(f"Unsupported state_policy: {self.state_policy!r}.")
