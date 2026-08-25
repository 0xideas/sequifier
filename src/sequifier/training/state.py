from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class TrainingState:
    phase_index: int = 0
    phase_epoch: int = 0
    phase_epoch_complete: bool = False
    source_index: int = 0
    source_scheduler_state: dict = field(default_factory=dict)
    iterator_positions: dict[str, int] = field(default_factory=dict)
    epoch: int = 0
    batch: int = 0
    global_batch_step: int = 0
    optimizer_step: int = 0
    accumulation_index: int = 0
    best_validation_loss: float = float("inf")
    epochs_without_improvement: int = 0
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
