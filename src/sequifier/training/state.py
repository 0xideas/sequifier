"""Explicit, checkpointable training progress."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class TrainingState:
    epoch: int = 0
    batch: int = 0
    global_batch_step: int = 0
    optimizer_step: int = 0
    accumulation_index: int = 0
    best_validation_loss: float = float("inf")
    epochs_without_improvement: int = 0
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
