"""Explicit capabilities for local PPO/GRPO environments and seeded rollout bridges."""

import json
from dataclasses import dataclass, field
from importlib import import_module
from typing import Literal, Mapping, Protocol

from sequifier.io.values import DomainValue, numeric_scalar


@dataclass(frozen=True)
class OnlineResetResult:
    observation: Mapping[str, DomainValue]
    static_inputs: Mapping[str, DomainValue] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OnlineStepResult:
    # None is permitted only at a true terminal, where no bootstrap is requested.
    observation: Mapping[str, DomainValue] | None
    reward: float
    terminated: bool = False
    truncated: bool = False
    units: Mapping[str, str] = field(default_factory=dict)
    success: bool | None = None


@dataclass(frozen=True)
class EnvironmentCapabilities:
    seeded_reset: bool = False
    exact_snapshot: bool = False
    group_start: Literal["snapshot", "scenario"] | None = None
    reproducible_group_start: bool = False
    # captured: the start token fixes simulator RNG state; reset_group must still
    # use its rollout seed for independent future environment randomness.
    group_start_randomness: Literal["captured", "independent"] = "independent"

    def __post_init__(self):
        if any(
            (not isinstance(flag, bool))
            for flag in (
                self.seeded_reset,
                self.exact_snapshot,
                self.reproducible_group_start,
            )
        ):
            raise ValueError("environment capability flags must be booleans")
        if self.group_start not in {
            None,
            "snapshot",
            "scenario",
        } or self.group_start_randomness not in {"captured", "independent"}:
            raise ValueError("unsupported group-start capability")
        if self.reproducible_group_start and self.group_start is None:
            raise ValueError(
                "reproducible_group_start requires a declared group_start capability"
            )


class OnlineEnvironmentAdapter(Protocol):
    protocol_version: int
    implementation_id: str
    capabilities: EnvironmentCapabilities

    def reset(self, seed: int) -> OnlineResetResult: ...
    def reset_from_observation(
        self, initial: OnlineResetResult, seed: int
    ) -> OnlineResetResult: ...
    def step(self, action: Mapping[str, DomainValue]) -> OnlineStepResult: ...
    def close(self) -> None: ...
    def snapshot_state(self) -> dict: ...
    def restore_state(self, state: dict) -> None: ...
    def create_group_start(self, seed: int) -> object: ...
    def reset_group(self, token: object, rollout_seed: int) -> OnlineResetResult: ...


def load_online_adapter(spec):
    module, attribute = spec.factory.split(":")
    adapter = getattr(import_module(module), attribute)(**spec.config)
    try:
        validate_online_adapter(adapter, spec)
        return adapter
    except BaseException:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
        raise


def validate_online_adapter(adapter, spec):
    if getattr(adapter, "protocol_version", None) != 3:
        raise ValueError("online adapters must explicitly declare protocol_version = 3")
    if getattr(adapter, "implementation_id", None) != spec.implementation_id:
        raise ValueError(
            "environment.implementation_id differs from adapter implementation"
        )
    if not isinstance(getattr(adapter, "capabilities", None), EnvironmentCapabilities):
        raise ValueError("online adapter requires explicit EnvironmentCapabilities")
    for method in ("reset", "step", "close"):
        if not callable(getattr(adapter, method, None)):
            raise ValueError(f"online adapter requires {method}")


def require_exact_snapshot(adapter):
    if adapter.capabilities.exact_snapshot is not True:
        raise ValueError("exact PPO resume requires capabilities.exact_snapshot")
    for method in ("snapshot_state", "restore_state"):
        if not callable(getattr(adapter, method, None)):
            raise ValueError(f"exact snapshot adapter requires {method}")


def require_group_start(adapter, *, exact=False):
    if adapter.capabilities.group_start not in {"snapshot", "scenario"}:
        raise ValueError(
            "GRPO requires explicit group_start: snapshot or scenario capability; a repeated seed is insufficient"
        )
    for method in ("create_group_start", "reset_group"):
        if not callable(getattr(adapter, method, None)):
            raise ValueError(f"group-start adapter requires {method}")
    if exact and adapter.capabilities.reproducible_group_start is not True:
        raise ValueError(
            "exact GRPO resume requires documented reproducible_group_start capability"
        )


def validate_start_token(token):
    try:
        # Tokens are opaque to Sequifier, but must survive portable JSON round trips.
        encoded = json.dumps(token, allow_nan=False, sort_keys=True)
        decoded = json.loads(encoded)
        if decoded != token:
            raise ValueError("group-start token changes after JSON serialization")
        return decoded
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "group-start token must be finite JSON-serializable content"
        ) from exc


def validate_reset(result, units):
    if not isinstance(result, OnlineResetResult) or dict(result.units) != units:
        raise ValueError(
            "online reset requires an initial observation with the frozen units"
        )
    return result


def validate_step(result, units):
    if not isinstance(result, OnlineStepResult):
        raise ValueError(
            "online step must return OnlineStepResult with an immediate scalar reward"
        )
    if not isinstance(result.terminated, bool) or not isinstance(
        result.truncated, bool
    ):
        raise ValueError("online boundary flags must be booleans")
    if dict(result.units) != units:
        raise ValueError("online observation units changed")
    numeric_scalar(result.reward, "online reward")
    if result.observation is None and not result.terminated:
        raise ValueError(
            "bootstrap requires the genuine successor, never an autoreset observation"
        )
    if result.success is not None and not isinstance(result.success, bool):
        raise ValueError("adapter success must be boolean when defined")
    return result


def require_seeded_reset(adapter):
    if not adapter.capabilities.seeded_reset or not callable(
        getattr(adapter, "reset_from_observation", None)
    ):
        raise ValueError(
            "policy rollout requires seeded_reset and reset_from_observation(initial, seed)"
        )


def adapter_identity(adapter):
    from dataclasses import asdict

    return {
        "module": type(adapter).__module__,
        "class": type(adapter).__qualname__,
        "protocol_version": adapter.protocol_version,
        "implementation_id": adapter.implementation_id,
        "capabilities": asdict(adapter.capabilities),
    }
