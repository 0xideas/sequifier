"""Sequential same-start groups with independent action/environment RNG streams."""

import json
from dataclasses import replace

import torch

from sequifier.fingerprints import fingerprint
from sequifier.rl.collection import PolicyCursor, stream_seed
from sequifier.rl.environment import require_group_start, validate_start_token
from sequifier.rl.grpo_buffer import GRPOTrajectory, SealedGroupBuffer
from sequifier.rl.history import frame_storage_bytes


class GroupCollectionError(RuntimeError):
    def __init__(self, message, executed_actions):
        super().__init__(
            f"{message}; discarded {executed_actions} uncommitted environment actions"
        )
        self.executed_actions = executed_actions


@torch.no_grad()
def collect_groups(
    context, adapter, start_group, count, generation, env_steps, restart_stream=0
):
    actor, reference, phase = context.actor.eval(), context.reference, context.phase
    require_group_start(adapter)
    bound = count * phase.group_size * phase.max_steps
    if phase.max_env_steps is not None and env_steps + bound > phase.max_env_steps:
        raise ValueError(
            "GRPO cannot fit the requested complete groups within max_env_steps (conservative group_size * max_steps bound)"
        )
    estimate = bound * (2048 + 128 * len(actor.interface.input_columns))
    if estimate > phase.max_buffer_bytes:
        raise ValueError(
            f"GRPO complete groups require approximately {estimate} buffer bytes"
        )
    groups, executed, used = [], 0, 0
    seed = stream_seed(
        context.run.seed, f"grpo-phase-{context.index}-restart-{restart_stream}", 0
    )
    try:
        for group_index in range(start_group, start_group + count):
            token = validate_start_token(
                adapter.create_group_start(
                    stream_seed(seed, "group-start", group_index)
                )
            )
            used += len(json.dumps(token).encode()) * phase.group_size + 1024
            trajectories, initial_key = [], None
            for sample in range(phase.group_size):
                # All samples restore the same opaque condition, then independently seed future randomness.
                cursor = PolicyCursor(
                    actor,
                    adapter,
                    context.run.units,
                    stream_seed(seed, f"group-{group_index}-actions", sample),
                )
                episode_id = f"group-{group_index}-sample-{sample}"
                cursor.reset(
                    adapter.reset_group(
                        validate_start_token(token),
                        stream_seed(seed, f"group-{group_index}-environment", sample),
                    ),
                    episode_id,
                )
                inputs = cursor.inputs
                key = fingerprint(
                    {
                        "features": {k: v.tolist() for k, v in inputs.inputs.items()},
                        "metadata": {k: v.tolist() for k, v in inputs.metadata.items()},
                        "units": dict(cursor.units),
                    }
                )
                if initial_key is not None and key != initial_key:
                    raise ValueError(
                        "GRPO samples have different canonical initial histories or units"
                    )
                initial_key = key
                frames, actions, rewards, old_scores, ref_scores = [], [], [], [], []
                reason = "max_steps"
                for step in range(phase.max_steps):
                    frame = cursor.history.current
                    used += frame_storage_bytes(frame) + 1024
                    if used > phase.max_buffer_bytes:
                        raise ValueError("GRPO group exceeded max_buffer_bytes")
                    inputs = cursor.inputs
                    decision = cursor.decide()
                    reference_score = float(
                        reference.evaluate_actions(
                            inputs,
                            torch.tensor(
                                [decision.action], dtype=torch.long, device=actor.device
                            ),
                        )[0]
                    )
                    executed += 1
                    step_result = cursor.step(decision)
                    reward = float(step_result.reward) * phase.reward_scale
                    if not torch.isfinite(
                        torch.tensor(
                            [reward, reference_score, decision.log_probability],
                            dtype=torch.float32,
                        )
                    ).all():
                        raise ValueError("non-finite GRPO reward or fixed action score")
                    frames.append(frame)
                    actions.append(decision.action)
                    rewards.append(reward)
                    old_scores.append(decision.log_probability)
                    ref_scores.append(reference_score)
                    if step_result.terminated or step_result.truncated:
                        reason = "terminated" if step_result.terminated else "truncated"
                        break
                score = float(torch.tensor(rewards, dtype=torch.float32).sum())
                if not torch.isfinite(torch.tensor(score)):
                    raise ValueError("non-finite GRPO trajectory score")
                trajectories.append(
                    GRPOTrajectory(
                        group_index,
                        sample,
                        token,
                        episode_id,
                        tuple(frames),
                        actor.config.context_length,
                        tuple(actions),
                        tuple(rewards),
                        tuple(old_scores),
                        tuple(ref_scores),
                        reason,
                        score,
                        0.0,
                        generation,
                    )
                )
            scores = torch.tensor([t.score for t in trajectories], dtype=torch.float32)
            mean, spread = scores.mean(), scores.std(unbiased=False)
            if not torch.isfinite(mean) or not torch.isfinite(spread):
                raise ValueError(
                    "non-finite FP32 GRPO group mean or population standard deviation"
                )
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
