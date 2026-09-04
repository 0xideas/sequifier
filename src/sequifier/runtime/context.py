from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class ExecutionEnvironment:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    distributed: bool


@dataclass(frozen=True)
class RunContext:
    project_root: Path
    model_name: str
    run_id: str
    session_id: str
    rank: int
    world_size: int
    logger: Any
