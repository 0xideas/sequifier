"""Common phase ordering, commit boundaries, and plan stopping."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar
from uuid import uuid4

Phase = TypeVar("Phase")
Result = TypeVar("Result")
NORMAL_COMPLETION = "normal_completion"


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


def cadence_due(progress: int, interval: int | None, *, final: bool) -> bool:
    return final or (interval is not None and progress % interval == 0)


def timed_checkpoints_due(
    strategy,
    *,
    now: float,
    snapshot_minutes: float | None,
    latest_minutes: float | None,
    last_snapshot: float,
    last_latest: float,
) -> tuple[bool, bool]:
    """Use the writer's wall clock for both kinds of timed checkpoint."""
    if snapshot_minutes is None and latest_minutes is None:
        return False, False
    due = (
        snapshot_minutes is not None and now - last_snapshot >= 60 * snapshot_minutes,
        latest_minutes is not None and now - last_latest >= 60 * latest_minutes,
    )
    return strategy.broadcast(due)


@dataclass(frozen=True)
class PhaseTransition(Generic[Phase]):
    index: int
    phase: Phase
    previous: Phase | None
    continues_optimizer: bool = False
    preserves_optimizer: bool = False


def validate_continuation(previous, phase, *, resolved=False):
    if phase.optimizer_state != "continue":
        return
    if previous is None:
        raise ValueError("the first phase cannot continue optimizer state")
    if previous.algorithm != phase.algorithm or previous.handoff != "last":
        raise ValueError(
            "optimizer continuation requires the same algorithm and a last handoff"
        )
    if (
        resolved
        and phase.algorithm != "supervised"
        and previous.model_interface != phase.model_interface
    ):
        raise ValueError("optimizer continuation cannot change model_interface")


def continuation_groups(phases, *, group_supervised=False):
    """Group optimizer continuations, optionally sharing a supervised runtime across resets."""
    start = 0
    for index, phase in enumerate(phases[1:], 1):
        supervised_pair = (
            group_supervised
            and phase.algorithm == phases[index - 1].algorithm == "supervised"
        )
        if phase.optimizer_state != "continue" and not supervised_pair:
            yield start, phases[start:index]
            start = index
    yield start, phases[start:]


def run_phases(
    phases: Sequence[Phase],
    run_phase: Callable[[PhaseTransition[Phase]], PhaseOutcome[Result]],
    commit_phase: Callable[[PhaseTransition[Phase], PhaseOutcome[Result]], None],
    *,
    start: int = 0,
    stopped: bool = False,
    completion_reason: str = NORMAL_COMPLETION,
    replay: Callable[[], None] | None = None,
    terminal_reason: str | None = None,
    optimizer_continues: Callable[[Phase], bool] | None = None,
) -> str:
    """Replay committed publications, then advance only after a durable commit."""
    if replay is not None:
        replay()
    if stopped:
        return "early_stopping"
    if terminal_reason is not None:
        return terminal_reason
    for index in range(start, len(phases)):
        transition = PhaseTransition(
            index,
            phases[index],
            phases[index - 1] if index else None,
            continues_optimizer=(
                index > 0
                and optimizer_continues is not None
                and optimizer_continues(phases[index])
            ),
            preserves_optimizer=(
                index + 1 < len(phases)
                and optimizer_continues is not None
                and optimizer_continues(phases[index + 1])
            ),
        )
        outcome = run_phase(transition)
        commit_phase(transition, outcome)
        if outcome.stop_scope == "end_plan":
            return "early_stopping"
        if outcome.completion_reason is not None:
            return outcome.completion_reason
    return completion_reason
