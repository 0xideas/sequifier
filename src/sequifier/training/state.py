from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field


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

    def snapshot(self) -> dict:
        """Return an independent, serializable rollback checkpoint."""
        return deepcopy(asdict(self))

    def restore(self, snapshot: dict) -> None:
        """Restore a snapshot produced by :meth:`snapshot`."""
        restored = TrainingState(**snapshot)
        self.phase_index = restored.phase_index
        self.phase_epoch = restored.phase_epoch
        self.phase_epoch_complete = restored.phase_epoch_complete
        self.source_index = restored.source_index
        self.source_scheduler_state = restored.source_scheduler_state
        self.iterator_positions = restored.iterator_positions
        self.epoch = restored.epoch
        self.batch = restored.batch
        self.global_batch_step = restored.global_batch_step
        self.optimizer_step = restored.optimizer_step
        self.accumulation_index = restored.accumulation_index
        self.best_validation_loss = restored.best_validation_loss
        self.epochs_without_improvement = restored.epochs_without_improvement
        self.run_id = restored.run_id
        self.session_id = restored.session_id
