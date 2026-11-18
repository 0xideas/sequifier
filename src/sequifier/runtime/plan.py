"""Common phase ordering, commit boundaries, and plan stopping."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar
from uuid import uuid4

Phase = TypeVar("Phase")
Result = TypeVar("Result")


def new_run_id() -> str:
    return uuid4().hex


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    session_id: str

    @classmethod
    def start(cls, run_id: str | None = None):
        return cls(run_id or new_run_id(), new_run_id())


@dataclass(frozen=True)
class PhaseOutcome(Generic[Result]):
    result: Result
    stop_scope: str | None = None
    completion_reason: str | None = None


def stop_scope(stopping, triggered: bool) -> str | None:
    return stopping.on_trigger if triggered and stopping is not None else None


def metric_improved(value, best, direction, min_delta=0) -> bool:
    if best is None:
        return True
    return value < best - min_delta if direction == "min" else value > best + min_delta


def select_handoff(handoff: str, last: Result, best: Result | None) -> Result:
    return best if handoff == "best" and best is not None else last


def execute_phase(
    index: int,
    phase: Phase,
    run_phase: Callable[[int, Phase], PhaseOutcome[Result]],
    commit_phase: Callable[[int, Phase, PhaseOutcome[Result]], None],
) -> PhaseOutcome[Result]:
    outcome = run_phase(index, phase)
    commit_phase(index, phase, outcome)
    return outcome


def run_phases(
    phases: Sequence[Phase],
    run_phase: Callable[[int, Phase], PhaseOutcome[Result]],
    commit_phase: Callable[[int, Phase, PhaseOutcome[Result]], None],
    *,
    start: int = 0,
    stopped: bool = False,
    completion_reason: str = "completed",
) -> str:
    """Commit each native result before acting on its stop or advancing."""
    if stopped:
        return "early_stopping"
    for index in range(start, len(phases)):
        outcome = execute_phase(index, phases[index], run_phase, commit_phase)
        if outcome.stop_scope == "end_plan":
            return "early_stopping"
        if outcome.completion_reason is not None:
            return outcome.completion_reason
    return completion_reason
