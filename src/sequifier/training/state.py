"""Serializable run lifecycle state, independent of the model."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from sequifier.artifacts.phase import CommittedPhases
from sequifier.artifacts.reports import HandoffCandidate
from sequifier.runtime.plan import new_run_id


@dataclass(frozen=True)
class RunStateSnapshot:
    values: dict[str, Any]
    committed_count: int


@dataclass
class RunState:
    phase_commits: CommittedPhases
    run_id: str = field(default_factory=new_run_id)
    session_id: str = field(default_factory=new_run_id)
    stopping_state: dict[str, Any] = field(default_factory=dict)
    last_validation_metric: float | None = None
    selection: HandoffCandidate = field(default_factory=HandoffCandidate)
    plan_stopped: bool = False
    initial_evaluated_phases: list[int] = field(default_factory=list)
    phase_index: int = 0  # Position within the native plan.
    metric_positions: dict = field(default_factory=dict)
    phase_offset: int = 0  # Original index of the first phase in the plan.
    phase_epoch: int = 0
    phase_epoch_complete: bool = False
    source_index: int = 0
    source_scheduler_state: dict[str, Any] = field(default_factory=dict)
    iterator_positions: dict[str, int] = field(default_factory=dict)
    epoch: int = 0
    batch: int = 0
    last_snapshot_batch_step: int = 0
    global_batch_step: int = 0
    optimizer_step: int = 0
    accumulation_index: int = 0
    backbone_parent_revision_id: str | None = None

    def snapshot(self) -> RunStateSnapshot:
        values = {
            key: value for key, value in vars(self).items() if key != "phase_commits"
        }
        return RunStateSnapshot(deepcopy(values), len(self.phase_commits))

    def restore(self, snapshot: RunStateSnapshot) -> None:
        if snapshot.committed_count != len(self.phase_commits):
            raise ValueError("Cannot roll back across a committed phase boundary")
        self.__dict__.update(deepcopy(snapshot.values))

    def state_dict(self) -> dict[str, Any]:
        values = self.snapshot().values
        values["selection"] = vars(values["selection"])
        values["phase_commits"] = self.phase_commits.records()
        return values

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any], phases) -> "RunState":
        fields = cls.__dataclass_fields__
        unknown = set(state).difference(fields)
        if unknown:
            raise ValueError(f"Unknown run-state fields: {sorted(unknown)!r}.")
        values = deepcopy(dict(state))
        values["selection"] = HandoffCandidate(**values["selection"])
        values["phase_commits"] = CommittedPhases.load(
            phases, values["phase_commits"], offset=values["phase_offset"]
        )
        return cls(**values)
