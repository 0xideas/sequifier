"""Bounded causal state shared by online collection and policy execution."""

import json
from collections import deque
from copy import deepcopy
from dataclasses import dataclass

from sequifier.io.policy_window import PolicyWindowView


@dataclass(frozen=True)
class PolicyFrame:
    observation: tuple
    previous_action: int | dict | None
    static_inputs: tuple
    episode_id: str
    step_id: int
    desired_return: float | None = None

    def state_dict(self):
        return {
            "observation": dict(self.observation),
            "previous_action": self.previous_action,
            "static_inputs": dict(self.static_inputs),
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "desired_return": self.desired_return,
        }

    @classmethod
    def from_state_dict(cls, values):
        return cls(
            tuple(values["observation"].items()),
            values["previous_action"],
            tuple(values["static_inputs"].items()),
            values["episode_id"],
            values["step_id"],
            values["desired_return"],
        )


def frame_storage_bytes(frame):
    """Conservative host-object allowance plus the complete domain payload."""
    return 1024 + 4 * len(
        json.dumps(frame.state_dict(), allow_nan=False).encode("utf-8")
    )


def history_inputs(spec, frames):
    frames = list(frames)
    return PolicyWindowView(spec).inputs(
        [dict(frame.observation) for frame in frames],
        [frame.previous_action for frame in frames],
        [frame.desired_return for frame in frames] if spec.return_conditioned else None,
        dict(frames[-1].static_inputs),
    )


class PolicyHistory:
    def __init__(self, spec, *, strict_vocabulary=False):
        self.spec = spec
        self.strict_vocabulary = strict_vocabulary
        self.frames = deque(maxlen=spec.context_length)
        self.reset_count = 0

    @property
    def attention_valid_mask(self):
        return (False,) * (self.spec.context_length - len(self.frames)) + (True,) * len(
            self.frames
        )

    @property
    def current(self):
        if not self.frames:
            raise ValueError("policy history has not been reset")
        return self.frames[-1]

    def _frame(
        self, observation, previous_action, static, episode_id, step_id, desired_return
    ):
        if (
            not isinstance(episode_id, str)
            or not episode_id
            or not isinstance(step_id, int)
            or isinstance(step_id, bool)
            or step_id < 0
        ):
            raise ValueError(
                "policy frame requires an episode ID and nonnegative integer step"
            )
        if set(observation) != set(self.spec.observation_columns) or set(static) != set(
            self.spec.static_columns
        ):
            raise ValueError(
                "environment payload does not match the model input binding"
            )
        frame = PolicyFrame(
            tuple(deepcopy(observation).items()),
            deepcopy(previous_action),
            tuple(deepcopy(static).items()),
            episode_id,
            step_id,
            desired_return,
        )
        history_inputs(self.spec, [frame])
        return frame

    def reset(self, observation, static_inputs, episode_id, desired_return=None):
        frame = self._frame(
            observation,
            None,
            static_inputs,
            episode_id,
            0,
            desired_return,
        )
        self.frames.clear()
        self.frames.append(frame)
        self.reset_count += 1

    def advance(self, action, successor, desired_return=None):
        current = self.current
        frame = self._frame(
            successor,
            action,
            dict(current.static_inputs),
            current.episode_id,
            current.step_id + 1,
            desired_return,
        )
        self.frames.append(frame)

    def inputs(self):
        return history_inputs(self.spec, self.frames)

    def state_dict(self):
        return {
            "frames": [frame.state_dict() for frame in self.frames],
            "reset_count": self.reset_count,
            "attention_valid_mask": self.attention_valid_mask,
        }

    def load_state_dict(self, values):
        frames = values["frames"]
        if not 0 < len(frames) <= self.spec.context_length:
            raise ValueError("invalid checkpoint history length")
        restored = []
        for values_frame in frames:
            frame = PolicyFrame.from_state_dict(values_frame)
            restored.append(
                self._frame(
                    dict(frame.observation),
                    frame.previous_action,
                    dict(frame.static_inputs),
                    frame.episode_id,
                    frame.step_id,
                    frame.desired_return,
                )
            )
        if any(
            right.episode_id != left.episode_id
            or right.step_id != left.step_id + 1
            or right.static_inputs != left.static_inputs
            for left, right in zip(restored, restored[1:])
        ):
            raise ValueError("checkpoint causal history is discontinuous")
        self.frames.clear()
        self.frames.extend(restored)
        self.reset_count = int(values["reset_count"])
        if (
            self.reset_count < 1
            or tuple(values["attention_valid_mask"]) != self.attention_valid_mask
        ):
            raise ValueError("checkpoint history mask or reset count is invalid")
