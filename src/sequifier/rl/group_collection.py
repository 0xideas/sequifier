"""Collection budgets, trajectory ownership and sealing shared by GRPO schedulers."""

from dataclasses import dataclass, field, replace

import torch

from sequifier.fingerprints import fingerprint
from sequifier.io.image_feature_cache import IMAGE_CACHE_PREFIX
from sequifier.rl.collection import PolicyCursor, stream_seed
from sequifier.rl.environment import OnlineStepResult, validate_start_token
from sequifier.rl.grpo_buffer import GRPOTrajectory, SealedGroupBuffer
from sequifier.rl.history import _json_payload_bytes


class GroupCollectionError(RuntimeError):
    def __init__(self, message, executed_actions):
        super().__init__(
            f"{message}; discarded {executed_actions} uncommitted environment actions"
        )
        self.executed_actions = executed_actions


@dataclass
class GroupTrajectory:
    group: int
    sample: int
    token: object
    cursor: PolicyCursor
    frames: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    old_scores: list = field(default_factory=list)
    reference_scores: list = field(default_factory=list)

    def append(
        self, frame, decision, reference_score, result: OnlineStepResult, reward_scale
    ):
        reward = float(result.reward) * reward_scale
        if not torch.isfinite(
            torch.tensor(
                [reward, reference_score, decision.log_probability], dtype=torch.float32
            )
        ).all():
            raise ValueError("non-finite GRPO reward or fixed action score")
        self.frames.append(frame)
        self.actions.append(decision.action)
        self.rewards.append(reward)
        self.old_scores.append(decision.log_probability)
        self.reference_scores.append(reference_score)

    def seal(self, generation, result: OnlineStepResult):
        score = float(torch.tensor(self.rewards, dtype=torch.float32).sum())
        if not torch.isfinite(torch.tensor(score)):
            raise ValueError("non-finite FP32 GRPO trajectory score")
        return GRPOTrajectory(
            self.group,
            self.sample,
            self.token,
            self.cursor.history.current.episode_id,
            tuple(self.frames),
            self.cursor.actor.config.context_length,
            tuple(self.actions),
            tuple(self.rewards),
            tuple(self.old_scores),
            tuple(self.reference_scores),
            "terminated"
            if result.terminated
            else "truncated"
            if result.truncated
            else "max_steps",
            score,
            0.0,
            generation,
        )


class GroupCollection:
    def __init__(self, context, adapter, count, generation, env_steps, restart_stream):
        self.context, self.adapter = context, adapter
        self.phase, self.generation = context.phase, generation
        bound = count * self.phase.group_size * self.phase.max_steps
        if (
            self.phase.max_env_steps is not None
            and env_steps + bound > self.phase.max_env_steps
        ):
            raise ValueError(
                "complete GRPO groups exceed max_env_steps (conservative group_size * max_steps bound)"
            )
        if (
            bound * (2048 + 128 * len(context.actor.interface.input_columns))
            > self.phase.max_buffer_bytes
        ):
            raise ValueError("complete GRPO groups exceed the minimum buffer estimate")
        self.seed = stream_seed(
            context.run.seed, f"grpo-phase-{context.index}-restart-{restart_stream}", 0
        )
        self.used, self.executed = 0, 0
        self.initial = {}

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None and isinstance(error, Exception):
            raise GroupCollectionError(str(error), self.executed) from error

    def account(self, size):
        self.used += size
        if self.used > self.phase.max_buffer_bytes:
            raise ValueError("GRPO collection exceeded max_buffer_bytes")

    def start(self, group):
        token = validate_start_token(
            self.adapter.create_group_start(
                stream_seed(self.seed, "group-start", group)
            )
        )
        self.account(_json_payload_bytes(token) * self.phase.group_size + 1024)
        return token

    def trajectory(self, group, sample, token, *, deferred=False):
        cursor = PolicyCursor(
            self.context.actor,
            self.adapter,
            self.context.run.units,
            stream_seed(self.seed, f"group-{group}-actions", sample),
            defer_device_transfer=deferred,
        )
        return GroupTrajectory(group, sample, token, cursor)

    def environment_seed(self, group, sample):
        return stream_seed(self.seed, f"group-{group}-environment", sample)

    def validate_initial(self, trajectory):
        cursor = trajectory.cursor
        inputs = cursor.inputs
        static = set(cursor.actor.config.static_columns)
        # reset() has one valid position. Include the full shape while avoiding
        # expansion of context_length copies of padded image pixels.
        identity = fingerprint(
            {
                "features": {
                    key: {
                        "shape": list(value.shape),
                        "values": (value if key in static else value[:, -1:]).tolist(),
                    }
                    for key, value in inputs.inputs.items()
                },
                "metadata": {
                    key: value.tolist()
                    for key, value in inputs.metadata.items()
                    if not key.startswith(IMAGE_CACHE_PREFIX)
                },
                "units": dict(cursor.units),
            }
        )
        if (
            trajectory.group in self.initial
            and self.initial[trajectory.group] != identity
        ):
            raise ValueError(
                "GRPO group members have different canonical initial histories or units"
            )
        self.initial[trajectory.group] = identity

    def seal_group(self, trajectories):
        if len(trajectories) != self.phase.group_size:
            raise ValueError("GRPO cannot seal an incomplete group")
        trajectories = sorted(trajectories, key=lambda value: value.sample_index)
        scores = torch.tensor([t.score for t in trajectories], dtype=torch.float32)
        mean, spread = scores.mean(), scores.std(unbiased=False)
        if not torch.isfinite(mean) or not torch.isfinite(spread):
            raise ValueError(
                "non-finite FP32 GRPO group mean or population standard deviation"
            )
        advantages = (scores - mean) / (spread + self.phase.advantage_epsilon)
        if not torch.isfinite(advantages).all():
            raise ValueError("non-finite GRPO group advantages")
        return tuple(
            replace(t, advantage=float(a)) for t, a in zip(trajectories, advantages)
        )

    def finish(self, groups):
        return SealedGroupBuffer(
            groups,
            self.generation,
            self.phase.group_size,
            self.context.reference_identity,
            self.used,
        ), self.executed
