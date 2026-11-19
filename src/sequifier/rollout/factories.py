"""Persistent adapter loading; factories receive their config as keyword arguments."""

from importlib import import_module
from typing import cast

from sequifier.config.rollout_config import AdapterConfig
from sequifier.contract_versions import (
    FORECAST_ROLLOUT_PROTOCOL,
    ContractVersions,
    require_contract_version,
)
from sequifier.rollout.contracts import EnvironmentAdapter


def validate_adapter(adapter):
    if getattr(adapter, "protocol", None) != FORECAST_ROLLOUT_PROTOCOL:
        raise ValueError(f"adapter.protocol must be {FORECAST_ROLLOUT_PROTOCOL!r}")
    protocol_version = getattr(adapter, "protocol_version", None)
    require_contract_version(
        "forecast rollout adapter protocol",
        protocol_version,
        ContractVersions.FORECAST_ROLLOUT_ADAPTER,
    )
    for method in ("reset", "step", "close"):
        if not callable(getattr(adapter, method, None)):
            raise ValueError(f"rollout adapter requires a callable {method} method")


def load_adapter(spec: AdapterConfig) -> EnvironmentAdapter:
    module, attribute = spec.factory.split(":")
    factory = getattr(import_module(module), attribute)
    if not callable(factory):
        raise ValueError(f"Adapter factory {spec.factory!r} is not callable")
    # Runtime validation happens inside collect_rollout's cleanup boundary.
    return cast(EnvironmentAdapter, factory(**spec.config))
