"""Version-2 online local adapter contract, independent of seeded v1 rollouts."""

from dataclasses import dataclass, field
from importlib import import_module
from typing import Mapping, Protocol

from sequifier.io.values import DomainValue, numeric_scalar
from sequifier.rollout.contracts import ResetRequest, StepRequest


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


class OnlineEnvironmentAdapter(Protocol):
    protocol_version: int
    supports_exact_snapshot: bool

    def reset(self, seed: int) -> OnlineResetResult: ...
    def step(self, action: Mapping[str, DomainValue]) -> OnlineStepResult: ...
    def close(self) -> None: ...
    def snapshot_state(self) -> dict: ...
    def restore_state(self, state: dict) -> None: ...


def load_online_adapter(spec):
    module, attribute = spec.factory.split(":")
    adapter = getattr(import_module(module), attribute)(**spec.config)
    try:
        if (
            not isinstance(getattr(adapter, "protocol_version", None), int)
            or adapter.protocol_version != 2
        ):
            raise ValueError(
                "online adapters must explicitly declare protocol_version = 2"
            )
        for method in ("reset", "step", "close"):
            if not callable(getattr(adapter, method, None)):
                raise ValueError(f"online adapter requires {method}")
        return adapter
    except BaseException:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
        raise


def require_exact_snapshot(adapter):
    if getattr(adapter, "supports_exact_snapshot", None) is not True:
        raise ValueError(
            "exact resume requires an explicit supports_exact_snapshot capability"
        )
    if not callable(getattr(adapter, "snapshot_state", None)) or not callable(
        getattr(adapter, "restore_state", None)
    ):
        raise ValueError(
            "exact snapshot adapters require snapshot_state and restore_state"
        )


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


class SeededAdapterBridge:
    """Expose a v1 adapter online when an explicit reset-state factory is available.

    The seed factory takes a seed and returns OnlineResetResult. All observations
    must be produced by environment bindings with identical names. Snapshot
    support is deliberately not inferred from the wrapped object.
    """

    protocol_version = 2
    supports_exact_snapshot = False

    def __init__(
        self, adapter, seed_factory, action_column, context_length=1, units=None
    ):
        from sequifier.rollout.factories import validate_adapter

        validate_adapter(adapter)
        if (
            not isinstance(context_length, int)
            or isinstance(context_length, bool)
            or context_length < 1
        ):
            raise ValueError("bridge context_length must be positive")
        self.adapter = adapter
        self.seed_factory = seed_factory
        self.action_column = action_column
        self.context_length = context_length
        self.units = units or {}
        self.resets = 0

    def reset(self, seed):
        result = validate_reset(self.seed_factory(seed), self.units)
        self.episode_id = f"bridge-{self.resets}"
        self.resets += 1
        self.step_id = 0
        self.observation = dict(result.observation)
        self.static_inputs = dict(result.static_inputs)
        self.adapter.reset(
            ResetRequest(
                self.episode_id,
                {
                    key: (value,) * self.context_length
                    for key, value in self.observation.items()
                },
                (False,) * (self.context_length - 1) + (True,),
                self.static_inputs,
                seed,
            )
        )
        return result

    def step(self, action):
        if set(action) != {self.action_column}:
            raise ValueError("bridge requires its one declared action column")
        result = self.adapter.step(
            StepRequest(
                self.episode_id,
                self.step_id,
                self.observation,
                dict(action),
                self.static_inputs,
            )
        )
        self.step_id += 1
        self.observation = dict(result.external_inputs)
        return OnlineStepResult(
            self.observation,
            result.reward,
            result.terminated,
            result.truncated,
            self.units,
        )

    def close(self):
        self.adapter.close()


def adapter_identity(adapter):
    return {
        "module": type(adapter).__module__,
        "class": type(adapter).__qualname__,
        "protocol_version": adapter.protocol_version,
        "supports_exact_snapshot": getattr(adapter, "supports_exact_snapshot", False)
        is True,
    }
