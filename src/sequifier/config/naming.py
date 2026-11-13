"""Shared configuration defaults and run-seed resolution."""

import secrets
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def resolve_run_seed(value: Any) -> int:
    if value == "random":
        return secrets.randbelow(1_000_000_000)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("seed must be an integer or 'random'")
    return value


RunSeed = Annotated[int, BeforeValidator(resolve_run_seed)]
