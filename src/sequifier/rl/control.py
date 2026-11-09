"""Minimal deterministic categorical control environment for local PPO examples.

The observation alternates between -1 and +1. Matching its sign earns +1;
choosing the other action earns -1. An episode lasts eight actions by default.
A random policy has expected return zero and the optimal return is the horizon.
"""

import random

from sequifier.rl.environment import (
    EnvironmentCapabilities,
    OnlineResetResult,
    OnlineStepResult,
)
from sequifier.rollout.contracts import StepResult


class CategoricalControlAdapter:
    protocol_version = 3
    implementation_id = "sequifier.categorical-control.v1"
    capabilities = EnvironmentCapabilities(
        exact_snapshot=True,
        group_start="scenario",
        reproducible_group_start=True,
        group_start_randomness="independent",
    )

    def __init__(self, horizon=8):
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
            raise ValueError("control horizon must be a positive integer")
        self.horizon = horizon
        self.step_id = 0
        self.cue = 1.0
        self.correct = 0
        self.ended = True

    def reset(self, seed):
        self.cue = random.Random(seed).choice((-1.0, 1.0))
        self.step_id, self.correct = 0, 0
        self.ended = False
        return OnlineResetResult({"cue": self.cue})

    def create_group_start(self, seed):
        return {"cue": random.Random(seed).choice((-1.0, 1.0)), "horizon": self.horizon}

    def reset_group(self, token, rollout_seed):
        if token.get("horizon") != self.horizon or token.get("cue") not in {-1.0, 1.0}:
            raise ValueError("incompatible control group-start token")
        self.step_id, self.correct, self.ended = 0, 0, False
        self.cue = token["cue"]
        return OnlineResetResult({"cue": self.cue})

    def step(self, action):
        if (
            self.ended
            or set(action) != {"action"}
            or action["action"] not in {"left", "right"}
        ):
            raise ValueError(
                "control requires a live episode and one left/right action"
            )
        correct = action["action"] == ("right" if self.cue > 0 else "left")
        self.correct += int(correct)
        self.step_id += 1
        self.cue = -self.cue
        self.ended = self.step_id == self.horizon
        return OnlineStepResult(
            {"cue": self.cue},
            1.0 if correct else -1.0,
            terminated=self.ended,
            success=self.correct == self.horizon if self.ended else None,
        )

    def snapshot_state(self):
        return {
            "horizon": self.horizon,
            "step_id": self.step_id,
            "cue": self.cue,
            "correct": self.correct,
            "ended": self.ended,
        }

    def restore_state(self, state):
        if (
            state["horizon"] != self.horizon
            or not 0 <= state["step_id"] <= self.horizon
            or state["cue"] not in {-1.0, 1.0}
            or not 0 <= state["correct"] <= state["step_id"]
            or not isinstance(state["ended"], bool)
        ):
            raise ValueError("incompatible control snapshot")
        self.step_id, self.cue = state["step_id"], state["cue"]
        self.correct, self.ended = state["correct"], state["ended"]

    def close(self):
        pass


class ControlRolloutAdapter:
    """Version-1 bridge for the existing sequifier infer rollout machinery."""

    protocol_version = 1

    def __init__(self, horizon=8):
        self.environment = CategoricalControlAdapter(horizon)

    def reset(self, request):
        self.environment.reset(request.seed)
        cue = request.seed_history["cue"][-1]
        if cue not in {-1.0, 1.0}:
            raise ValueError("control seed cue must be -1 or +1")
        self.environment.cue = float(cue)

    def step(self, request):
        if request.observation != {"cue": self.environment.cue}:
            raise ValueError("control observation and environment disagree")
        result = self.environment.step(request.predictions)
        if result.observation is None:
            raise ValueError("control must return the true successor")
        return StepResult(
            result.observation, result.reward, result.terminated, result.truncated
        )

    def close(self):
        self.environment.close()
