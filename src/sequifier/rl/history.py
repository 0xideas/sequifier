"""Bounded causal state shared by online collection and policy execution."""

import json
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch

from sequifier.io.image_feature_cache import (
    IMAGE_CACHE_PREFIX,
    image_feature_cache_enabled,
    new_image_identity,
)
from sequifier.io.policy_window import PolicyWindowView
from sequifier.io.values import portable_fields


@dataclass(frozen=True)
class PolicyFrame:
    observation: tuple
    previous_action: int | dict | None
    static_inputs: tuple
    episode_id: str
    step_id: int
    desired_return: float | None = None

    @cached_property
    def image_identities(self):
        # Transient IDs are deliberately excluded from state_dict and archives.
        return {
            key.removeprefix("__image:"): torch.tensor(
                [new_image_identity() for _ in value], dtype=torch.int64
            )
            for key, value in self.observation
            if key.startswith("__image:")
        }

    def _values(self):
        return {
            "observation": dict(self.observation),
            "previous_action": self.previous_action,
            "static_inputs": dict(self.static_inputs),
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "desired_return": self.desired_return,
        }

    def state_dict(self):
        # Materialize portable lists only at checkpoint/archive boundaries.
        return portable_fields(self._values())

    @cached_property
    def storage_bytes(self):
        return 1024 + 4 * _json_payload_bytes(self._values())

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


def _json_payload_bytes(value):
    """Size of default json.dumps(...).encode('utf-8') without expanding pixels."""
    if isinstance(value, np.ndarray) and value.ndim and value.size:
        if value.dtype == np.uint8:
            digits = (
                value.size
                + np.count_nonzero(value >= 10)
                + np.count_nonzero(value >= 100)
            )
        elif value.dtype == np.bool_:
            digits = 5 * value.size - np.count_nonzero(value)
        else:
            return _json_payload_bytes(value.tolist())
        # Each list contributes brackets; separators contribute two bytes.
        parents = sum(int(np.prod(value.shape[:axis])) for axis in range(1, value.ndim))
        return int(digits + 2 * value.size + 2 * parents)
    if isinstance(value, np.ndarray):
        return _json_payload_bytes(value.tolist())
    if isinstance(value, dict):
        return (
            2
            + max(0, len(value) - 1) * 2
            + sum(
                len(json.dumps({key: None}, allow_nan=False).encode("utf-8"))
                - 6
                + _json_payload_bytes(item)
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return (
            2
            + max(0, len(value) - 1) * 2
            + sum(_json_payload_bytes(item) for item in value)
        )
    return len(json.dumps(value, allow_nan=False).encode("utf-8"))


def frame_storage_bytes(frame):
    """Conservative host-object allowance plus the complete domain payload."""
    return frame.storage_bytes


def history_inputs(spec, frames):
    frames = list(frames)
    inputs = PolicyWindowView(spec).inputs(
        [dict(frame.observation) for frame in frames],
        [frame.previous_action for frame in frames],
        [frame.desired_return for frame in frames] if spec.return_conditioned else None,
        dict(frames[-1].static_inputs),
    )
    if image_feature_cache_enabled():
        for view in frames[-1].image_identities:
            identities = torch.stack([frame.image_identities[view] for frame in frames])
            padding = identities.new_zeros(
                spec.context_length - len(frames), *identities.shape[1:]
            )
            inputs.metadata[IMAGE_CACHE_PREFIX + view] = torch.cat(
                [padding, identities]
            )[None]
    return inputs


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
        owned = {}
        for key, value in observation.items():
            if key.startswith("__image:") or key.startswith("__image_mask:"):
                dtype = torch.uint8 if key.startswith("__image:") else torch.bool
                # Preserve the materializer's dtype conversion and take one
                # owned compact copy, instead of retaining nested pixel lists.
                owned[key] = torch.as_tensor(value, dtype=dtype).cpu().numpy().copy()
            else:
                owned[key] = deepcopy(value)
        frame = PolicyFrame(
            tuple(owned.items()),
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
