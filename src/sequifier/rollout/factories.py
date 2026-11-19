"""Persistent adapter loading; factories receive their config as keyword arguments."""

from typing import cast

from sequifier.config.rollout_config import AdapterConfig
from sequifier.contract_versions import (
    FORECAST_ROLLOUT_PROTOCOL,
    ContractVersions,
    require_contract_version,
)
from sequifier.integration.factories import load_factory
from sequifier.rollout.contracts import EnvironmentAdapter


def validate_adapter(adapter):
    try:
        protocol = adapter.protocol
    except AttributeError:
        protocol = None
    if protocol != FORECAST_ROLLOUT_PROTOCOL:
        raise ValueError(f"adapter.protocol must be {FORECAST_ROLLOUT_PROTOCOL!r}")
    try:
        protocol_version = adapter.protocol_version
    except AttributeError:
        protocol_version = None
    require_contract_version(
        "forecast rollout adapter protocol",
        protocol_version,
        ContractVersions.FORECAST_ROLLOUT_ADAPTER,
    )
    try:
        methods = (adapter.reset, adapter.step, adapter.close)
    except AttributeError as error:
        raise ValueError(
            f"rollout adapter requires a callable {error.name} method"
        ) from error
    for name, method in zip(("reset", "step", "close"), methods):
        if not callable(method):
            raise ValueError(f"rollout adapter requires a callable {name} method")


def load_adapter(spec: AdapterConfig) -> EnvironmentAdapter:
    factory = load_factory(spec.factory)
    if not callable(factory):
        raise ValueError(f"Adapter factory {spec.factory!r} is not callable")
    # Runtime validation happens inside collect_rollout's cleanup boundary.
    return cast(EnvironmentAdapter, factory(**spec.config))
