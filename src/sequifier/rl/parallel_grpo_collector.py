"""Explicitly enabled synchronous GRPO collection across independent simulators."""

import math
from collections import deque
from dataclasses import dataclass, field, replace

import torch

from sequifier.fingerprints import fingerprint
from sequifier.io.actions import action_record, decode_action
from sequifier.io.batch import collate_model_inputs
from sequifier.io.image_feature_cache import IMAGE_CACHE_PREFIX
from sequifier.rl.collection import PolicyCursor, PolicyDecision, stream_seed
from sequifier.rl.environment import validate_start_token
from sequifier.rl.grpo_buffer import GRPOTrajectory, SealedGroupBuffer
from sequifier.rl.grpo_collector import GroupCollectionError
from sequifier.rl.history import frame_storage_bytes
from sequifier.rl.simulator_workers import require_parallel_group_start


@dataclass
class _Trajectory:
    group: int
    sample: int
    token: object
    cursor: PolicyCursor
    frames: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    old_scores: list = field(default_factory=list)
    reference_scores: list = field(default_factory=list)

    def seal(self, generation, result):
        score = float(torch.tensor(self.rewards, dtype=torch.float32).sum())
        if not math.isfinite(score):
            raise ValueError("non-finite FP32 GRPO trajectory score")
        reason = (
            "terminated"
            if result.terminated
            else "truncated"
            if result.truncated
            else "max_steps"
        )
        return GRPOTrajectory(
            group_index=self.group,
            sample_index=self.sample,
            start_token=self.token,
            episode_id=self.cursor.history.current.episode_id,
            frames=tuple(self.frames),
            context_length=self.cursor.actor.config.context_length,
            actions=tuple(self.actions),
            rewards=tuple(self.rewards),
            old_log_probabilities=tuple(self.old_scores),
            reference_log_probabilities=tuple(self.reference_scores),
            cutoff_reason=reason,
            score=score,
            advantage=0.0,
            generation=generation,
        )


@torch.no_grad()
def _decide(actor, reference, cursors):
    inputs = collate_model_inputs([cursor.inputs for cursor in cursors]).to(
        actor.device
    )
    distribution = actor.distribution(inputs)
    generators = [cursor.generator for cursor in cursors]
    # Decode and validate distributions across the full environment batch.
    # Only random draws use individual generators; no per-row decoder calls.
    actions = distribution.select(False, generators)
    old_scores = distribution.log_prob(actions)
    scores = (
        torch.stack([old_scores, reference.evaluate_actions(inputs, actions)], dim=1)
        .cpu()
        .tolist()
    )
    host_actions = {name: value.cpu() for name, value in actions.items()}
    decisions = []
    for index, (old, ref) in enumerate(scores):
        if not math.isfinite(old) or not math.isfinite(ref):
            raise ValueError("non-finite GRPO policy/reference action score")
        action = action_record(host_actions, index)
        decisions.append(
            (
                PolicyDecision(
                    decode_action(action, actor.config), action, old, 0.0, 0.0
                ),
                ref,
            )
        )
    return decisions


def _initial_history_identity(cursor):
    # reset() has exactly one valid temporal position; all earlier positions
    # are canonical zero padding. Compare its encoded values and full shapes
    # without expanding context_length copies of padded camera pixels.
    inputs = cursor.inputs
    static = set(cursor.actor.config.static_columns)
    return fingerprint(
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


@torch.no_grad()
def collect_parallel_groups(
    context,
    adapter,
    workers,
    start_group,
    count,
    generation,
    env_steps,
    restart_stream,
):
    actor, reference, phase = (
        context.actor.eval(),
        context.reference.eval(),
        context.phase,
    )
    require_parallel_group_start(adapter)
    bound = count * phase.group_size * phase.max_steps
    if phase.max_env_steps is not None and env_steps + bound > phase.max_env_steps:
        raise ValueError("complete GRPO groups exceed max_env_steps")
    if (
        bound * (2048 + 128 * len(actor.interface.input_columns))
        > phase.max_buffer_bytes
    ):
        raise ValueError("complete GRPO groups exceed the minimum buffer estimate")
    seed = stream_seed(
        context.run.seed, f"grpo-phase-{context.index}-restart-{restart_stream}", 0
    )
    used, executed = 0, 0
    pending, active, completed, initial = deque(), {}, {}, {}

    def account(size):
        nonlocal used
        used += size
        if used > phase.max_buffer_bytes:
            raise ValueError("GRPO collection exceeded max_buffer_bytes")

    try:
        from sequifier.rl.history import _json_payload_bytes

        for group in range(start_group, start_group + count):
            token = validate_start_token(
                adapter.create_group_start(stream_seed(seed, "group-start", group))
            )
            account(_json_payload_bytes(token) * phase.group_size + 1024)
            completed[group] = []
            pending.extend((group, sample, token) for sample in range(phase.group_size))
        while pending or active:
            resetting = []
            for slot in range(len(workers)):
                if slot in active or not pending:
                    continue
                group, sample, token = pending.popleft()
                cursor = PolicyCursor(
                    actor,
                    adapter,
                    context.run.units,
                    stream_seed(seed, f"group-{group}-actions", sample),
                    defer_device_transfer=True,
                )
                active[slot] = _Trajectory(group, sample, token, cursor)
                workers.send(
                    slot,
                    "reset_group",
                    validate_start_token(token),
                    stream_seed(seed, f"group-{group}-environment", sample),
                )
                resetting.append(slot)
            for slot in resetting:
                state = active[slot]
                state.cursor.reset(
                    workers.receive(slot),
                    f"group-{state.group}-sample-{state.sample}",
                )
                frame = state.cursor.history.current
                identity = _initial_history_identity(state.cursor)
                if state.group in initial and initial[state.group] != identity:
                    raise ValueError(
                        "GRPO group members have different initial histories"
                    )
                initial[state.group] = identity
                # Include all live history before any environment step is issued.
                account(frame_storage_bytes(frame) + 1024)
                account(
                    sum(
                        value.numel() * value.element_size()
                        for value in (
                            *state.cursor.inputs.inputs.values(),
                            *state.cursor.inputs.metadata.values(),
                        )
                    )
                )
            decisions = {}
            slots = sorted(active)
            for start in range(0, len(slots), phase.rollout_inference_batch_size):
                chunk = slots[start : start + phase.rollout_inference_batch_size]
                scored = _decide(actor, reference, [active[s].cursor for s in chunk])
                decisions.update(zip(chunk, scored))
            for decision, _ in decisions.values():
                account(4 * _json_payload_bytes(decision.action))
            for slot, (decision, _) in decisions.items():
                # Count attempted actions conservatively before sending: a failed
                # send must never hide a potentially executed simulator action.
                executed += 1
                workers.send(slot, "step", decision.domain_action)
            for slot, (decision, ref) in decisions.items():
                state = active[slot]
                state.frames.append(state.cursor.history.current)
                result = state.cursor.accept_step(decision, workers.receive(slot))
                reward = float(result.reward) * phase.reward_scale
                if not math.isfinite(reward) or not torch.isfinite(
                    torch.tensor(reward, dtype=torch.float32)
                ):
                    raise ValueError("GRPO reward exceeds finite FP32 range")
                state.actions.append(decision.action)
                state.rewards.append(reward)
                state.old_scores.append(decision.log_probability)
                state.reference_scores.append(ref)
                # Account the successor retained by a live cursor, even if the
                # episode ends or the collector's finite horizon is reached.
                if result.observation is not None:
                    account(frame_storage_bytes(state.cursor.history.current) + 1024)
                if (
                    result.terminated
                    or result.truncated
                    or len(state.actions) == phase.max_steps
                ):
                    completed[state.group].append(state.seal(generation, result))
                    del active[slot]
        groups = []
        for trajectories in completed.values():
            if len(trajectories) != phase.group_size:
                raise ValueError("GRPO cannot seal an incomplete group")
            trajectories.sort(key=lambda value: value.sample_index)
            scores = torch.tensor([t.score for t in trajectories], dtype=torch.float32)
            mean, spread = scores.mean(), scores.std(unbiased=False)
            advantages = (scores - mean) / (spread + phase.advantage_epsilon)
            if not torch.isfinite(advantages).all():
                raise ValueError("non-finite GRPO group advantages")
            groups.append(
                tuple(
                    replace(t, advantage=float(a))
                    for t, a in zip(trajectories, advantages)
                )
            )
        return SealedGroupBuffer(
            groups, generation, phase.group_size, context.reference_identity, used
        ), executed
    except Exception as exc:
        raise GroupCollectionError(str(exc), executed) from exc
