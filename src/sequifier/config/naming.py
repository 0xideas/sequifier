"""Shared configuration defaults and run-seed resolution."""

import secrets
from typing import Annotated, Any, TypeAlias

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SeedInteger: TypeAlias = Annotated[int, Field(strict=True, ge=0, lt=2**32)]


def supervised_phase_seed(run_seed: int, index: int) -> int:
    return (run_seed + index) % 2**32


def resolve_run_seed(value: Any) -> int:
    if value == "random":
        return secrets.randbelow(1_000_000_000)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("seed must be an integer or 'random'")
    return value


RunSeed: TypeAlias = Annotated[SeedInteger, BeforeValidator(resolve_run_seed)]
