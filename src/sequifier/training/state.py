"""Serializable run lifecycle state, independent of the model."""

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from torch import Tensor


@dataclass(frozen=True)
class RunStateSnapshot:
    values: dict[str, Any]


@dataclass
class RunState:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    phase_index: int = 0
    phase_epoch: int = 0
    phase_epoch_complete: bool = False
    source_index: int = 0
    source_scheduler_state: dict[str, Any] = field(default_factory=dict)
    iterator_positions: dict[str, int] = field(default_factory=dict)
    epoch: int = 0
    batch: int = 0
    global_batch_step: int = 0
    optimizer_step: int = 0
    accumulation_index: int = 0
    best_validation_loss: float = float("inf")
    epochs_without_improvement: int = 0
    best_model_state_dict: dict[str, Tensor] | None = None
    backbone_parent_revision_id: str | None = None

    def snapshot(self) -> RunStateSnapshot:
        return RunStateSnapshot(deepcopy(self.state_dict()))

    def restore(self, snapshot: RunStateSnapshot) -> None:
        restored = RunState.from_state_dict(snapshot.values)
        self.__dict__.clear()
        self.__dict__.update(restored.__dict__)

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "RunState":
        fields = cls.__dataclass_fields__
        unknown = set(state).difference(fields)
        if unknown:
            raise ValueError(f"Unknown run-state fields: {sorted(unknown)!r}.")
        return cls(**deepcopy(dict(state)))
