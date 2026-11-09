"""Compact, single-generation on-policy storage and fixed FP32 GAE targets."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch

from sequifier.config.policy_config import CategoricalPolicyConfig
from sequifier.io.batch import SequifierInputs
from sequifier.io.tensors import collate_tensor_records
from sequifier.rl.history import frame_storage_bytes, history_inputs
from sequifier.rl.run_state import write_json_atomic


@dataclass(frozen=True)
class PPODecision:
    frame: int
    successor: int | None
    action: int
    domain_action: str | float | int
    old_log_probability: float
    old_value: float
    entropy: float
    reward: float
    raw_reward: float
    successor_value: float
    terminated: bool
    truncated: bool
    bootstrap: bool
    episode_id: str
    step_id: int
    policy_version: int


@dataclass(frozen=True)
class PPORolloutBuffer:
    """Sealed tuples retain each frame once, including the pre-buffer context.

    Access reconstructs complete ordered windows. No optimization tensor is
    shared with the fixed old scores or targets held by this buffer.
    """

    spec: CategoricalPolicyConfig
    policy_version: int
    frames: tuple
    decisions: tuple[PPODecision, ...]
    advantages: tuple[float, ...]
    targets: tuple[float, ...]
    estimated_bytes: int

    def __len__(self):
        return len(self.decisions)

    def require_version(self, version):
        if version != self.policy_version or any(
            step.policy_version != version for step in self.decisions
        ):
            raise ValueError("PPO cannot update from another policy generation")

    def window(self, reference):
        frame = self.frames[reference]
        start = reference
        while (
            start > 0
            and reference - start + 1 < self.spec.context_length
            and self.frames[start - 1].episode_id == frame.episode_id
        ):
            if self.frames[start - 1].step_id + 1 != self.frames[start].step_id:
                raise ValueError("PPO frame history is discontinuous")
            start -= 1
        return history_inputs(self.spec, self.frames[start : reference + 1])

    def batch(self, indices, device):
        indices = [int(index) for index in indices]
        return (
            cast(
                SequifierInputs,
                collate_tensor_records(
                    [self.window(self.decisions[index].frame) for index in indices]
                ),
            ).to(device),
            torch.tensor(
                [self.decisions[index].action for index in indices],
                dtype=torch.long,
                device=device,
            ),
            torch.tensor(
                [self.decisions[index].old_log_probability for index in indices],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [self.decisions[index].old_value for index in indices],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [self.advantages[index] for index in indices],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [self.targets[index] for index in indices],
                dtype=torch.float32,
                device=device,
            ),
        )

    def archive(self, path, *, include_scores):
        """A separate versioned collection record, never an offline update source."""
        decisions = []
        for index, step in enumerate(self.decisions):
            record = asdict(step)
            boundary = step.terminated or step.truncated
            cut = index == len(self) - 1 and not boundary
            record["cutoff_reason"] = (
                "terminated"
                if step.terminated
                else "truncated"
                if step.truncated
                else "collection_cut"
                if cut
                else None
            )
            if not include_scores:
                for key in (
                    "old_log_probability",
                    "old_value",
                    "entropy",
                    "successor_value",
                ):
                    record.pop(key)
            decisions.append(record)
        payload = {
            "format": "sequifier.ppo-collection",
            "version": 1,
            "policy_version": self.policy_version,
            "policy": self.spec.model_dump(mode="json"),
            "frames": [frame.state_dict() for frame in self.frames],
            "decisions": decisions,
        }
        write_json_atomic(path, payload)


def load_collection_record(path):
    """Read the independent archival format without changing v1 episode readers."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        payload.get("format") != "sequifier.ppo-collection"
        or payload.get("version") != 1
    ):
        raise ValueError("expected version-1 PPO collection record")
    from sequifier.config.policy_config import CategoricalPolicyConfig
    from sequifier.rl.history import PolicyFrame

    spec = CategoricalPolicyConfig.model_validate(payload["policy"])
    frames = [PolicyFrame.from_state_dict(frame) for frame in payload["frames"]]
    for frame in frames:
        history_inputs(spec, [frame])
    for step in payload["decisions"]:
        frame = frames[step["frame"]]
        if (
            step["policy_version"] != payload["policy_version"]
            or (step["episode_id"], step["step_id"])
            != (frame.episode_id, frame.step_id)
            or not 0 <= step["action"] < len(spec.action_values)
            or step["domain_action"] != spec.action_values[step["action"]]
            or step["bootstrap"] != (not step["terminated"])
        ):
            raise ValueError("collection record violates the decision contract")
        if step["successor"] is None:
            if step["bootstrap"]:
                raise ValueError("collection record lacks a bootstrap successor")
        else:
            successor = frames[step["successor"]]
            if (
                successor.episode_id != frame.episode_id
                or successor.step_id != frame.step_id + 1
                or successor.previous_action != step["action"]
            ):
                raise ValueError("collection record successor is discontinuous")
        if not torch.isfinite(torch.tensor(step["reward"], dtype=torch.float32)):
            raise ValueError("collection record reward must be finite FP32")
    return payload


def compute_gae(decisions, gamma, gae_lambda, normalize):
    """Boundary-aware FP32 targets; normalization happens once per iteration."""
    advantages = torch.zeros(len(decisions), dtype=torch.float32)
    following_advantage = torch.zeros((), dtype=torch.float32)
    for index in range(len(decisions) - 1, -1, -1):
        step = decisions[index]
        continues = (
            index + 1 < len(decisions)
            and not (step.terminated or step.truncated)
            and decisions[index + 1].episode_id == step.episode_id
            and decisions[index + 1].step_id == step.step_id + 1
        )
        delta = (
            torch.tensor(step.reward, dtype=torch.float32)
            + gamma
            * float(step.bootstrap)
            * torch.tensor(step.successor_value, dtype=torch.float32)
            - torch.tensor(step.old_value, dtype=torch.float32)
        )
        following_advantage = (
            delta + gamma * gae_lambda * float(continues) * following_advantage
        )
        advantages[index] = following_advantage
    targets = advantages + torch.tensor(
        [step.old_value for step in decisions], dtype=torch.float32
    )
    if normalize and len(decisions) > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(
            unbiased=False
        ).clamp_min(1e-8)
    if not torch.isfinite(advantages).all() or not torch.isfinite(targets).all():
        raise ValueError("non-finite PPO advantage or target")
    return tuple(advantages.tolist()), tuple(targets.tolist())


class PPOBufferBuilder:
    def __init__(self, spec, version, horizon, initial_frames, max_bytes):
        self.spec = spec.model_copy(deep=True)
        self.version = version
        self.horizon = horizon
        self.frames = list(initial_frames)
        self.decisions = []
        self.max_bytes = max_bytes
        self.bytes_per_frame = max(
            [frame_storage_bytes(frame) for frame in self.frames] + [1024]
        )
        self.estimated_bytes = (
            2 * horizon + spec.context_length
        ) * self.bytes_per_frame + horizon * 1024
        if self.estimated_bytes > max_bytes:
            raise ValueError(
                f"PPO buffer estimate {self.estimated_bytes} exceeds max_buffer_bytes={max_bytes}"
            )
        self.used_bytes = sum(frame_storage_bytes(frame) for frame in self.frames)
        self.sealed = False

    def append_frame(self, frame):
        if self.sealed:
            raise ValueError("PPO buffer is sealed")
        cost = frame_storage_bytes(frame)
        if self.used_bytes + cost + len(self.decisions) * 1024 > self.max_bytes:
            raise ValueError("PPO history exceeded its declared memory budget")
        if len(self.frames) >= 2 * self.horizon + self.spec.context_length:
            raise ValueError("PPO history exceeded its bounded frame capacity")
        self.frames.append(frame)
        self.used_bytes += cost
        return len(self.frames) - 1

    def append(self, decision):
        if self.sealed or len(self.decisions) >= self.horizon:
            raise ValueError("PPO buffer is sealed or full")
        if decision.policy_version != self.version:
            raise ValueError("policy generation changed during collection")
        self.decisions.append(decision)
        if self.used_bytes + len(self.decisions) * 1024 > self.max_bytes:
            raise ValueError("PPO decisions exceeded the memory budget")

    def seal(self, gamma, gae_lambda, normalize):
        if self.sealed or len(self.decisions) != self.horizon:
            raise ValueError("PPO must seal exactly one complete collection horizon")
        self.sealed = True
        advantages, targets = compute_gae(self.decisions, gamma, gae_lambda, normalize)
        return PPORolloutBuffer(
            self.spec,
            self.version,
            tuple(self.frames),
            tuple(self.decisions),
            advantages,
            targets,
            max(self.estimated_bytes, self.used_bytes + len(self.decisions) * 1024),
        )
