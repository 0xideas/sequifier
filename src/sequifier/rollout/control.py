"""Deterministic finite-horizon binary numeric control task."""

from sequifier.rollout.contracts import ResetRequest, StepRequest, StepResult
from sequifier.rollout.history import numeric_fields, numeric_scalar


class BinaryControlAdapter:
    """Action 0 holds position, action 1 increments it; reward equals the action."""

    protocol_version = 1

    def __init__(self, horizon: int = 3):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("control horizon must be a positive integer")
        self.horizon = horizon
        self.closed = False
        self.episode_id = None
        self.steps = 0
        self.position = 0.0

    def reset(self, request: ResetRequest) -> None:
        if self.closed:
            raise ValueError("control adapter is closed")
        observation = numeric_fields(
            {key: values[-1] for key, values in request.seed_history.items()},
            {"position", "remaining_steps"},
            "control seed",
        )
        if observation != {"position": 0.0, "remaining_steps": float(self.horizon)}:
            raise ValueError(
                "control episodes start at position=0 with the full horizon remaining"
            )
        self.episode_id = request.episode_id
        self.steps = 0
        self.position = 0.0

    def step(self, request: StepRequest) -> StepResult:
        if (
            self.closed
            or self.episode_id != request.episode_id
            or self.steps >= self.horizon
        ):
            raise ValueError("control step requires an active reset episode")
        if request.step_id != self.steps or request.observation != {
            "position": self.position,
            "remaining_steps": float(self.horizon - self.steps),
        }:
            raise ValueError("control request observation/step is out of sync")
        action = numeric_scalar(request.predictions.get("action"), "control action")
        if action not in (0.0, 1.0) or set(request.predictions) != {"action"}:
            raise ValueError("binary control actions must be 0 or 1")
        self.position += action
        self.steps += 1
        return StepResult(
            {
                "position": self.position,
                "remaining_steps": float(self.horizon - self.steps),
            },
            reward=action,
            terminated=self.steps == self.horizon,
        )

    def close(self) -> None:
        self.closed = True
