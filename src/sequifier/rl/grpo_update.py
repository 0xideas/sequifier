"""Trajectory-weighted clipped GRPO with a sampled frozen-reference penalty."""

import torch

from sequifier.rl.run_state import apply_gradients


def grpo_objective(
    log_probability, old_log_probability, reference_log_probability, advantage, phase
):
    log_ratio = log_probability.float() - old_log_probability.float()
    ratio = log_ratio.exp()
    reference_delta = reference_log_probability.float() - log_probability.float()
    penalty = torch.expm1(reference_delta) - reference_delta
    clipped = ratio.clamp(1 - phase.clip_epsilon, 1 + phase.clip_epsilon)
    loss = (
        -torch.minimum(ratio * advantage, clipped * advantage)
        + phase.kl_coefficient * penalty
    )
    old_kl = ratio - 1 - log_ratio
    if any(not torch.isfinite(t).all() for t in (ratio, penalty, loss, old_kl)):
        raise ValueError(
            "non-finite GRPO ratio, sampled reference penalty or objective"
        )
    return loss, penalty, old_kl, ((ratio - 1).abs() > phase.clip_epsilon).float()


@torch.no_grad()
def validate_old_scores(actor, buffer, chunk_size):
    for trajectory in buffer.trajectories:
        for inputs, actions, old, _ in trajectory.batches(actor, chunk_size):
            if not torch.allclose(
                actor.evaluate_actions(inputs, actions).float(),
                old,
                atol=1e-5,
                rtol=1e-4,
            ):
                raise ValueError(
                    "GRPO reconstructed old-policy scores differ before first update"
                )


def update_grpo(
    context, buffer, optimizer, scheduler, sampler, generation, optimizer_steps
):
    actor, phase = context.actor.eval(), context.phase
    buffer.require_fresh(generation, context.reference_identity)
    if actor.grpo_policy_version != generation:
        raise ValueError("GRPO collection generation differs from actor")
    validate_old_scores(actor, buffer, phase.scoring_chunk_size)
    trajectories = buffer.trajectories
    totals = {
        key: 0.0
        for key in (
            "loss",
            "sampled_reference_penalty",
            "old_policy_approximate_kl",
            "clip_fraction",
        )
    }
    scored, stopped = 0, False
    size = phase.batch_size * phase.accumulation_steps
    epoch = -1
    for epoch in range(phase.optimization_epochs):
        order = torch.randperm(len(trajectories), generator=sampler).tolist()
        for start in range(0, len(order), size):
            indices = order[start : start + size]
            if phase.old_policy_target_kl is not None:
                old_kl = 0.0
                with torch.no_grad():
                    for i in indices:
                        trajectory = trajectories[i]
                        for inputs, actions, old, ref in trajectory.batches(
                            actor, phase.scoring_chunk_size
                        ):
                            _, _, kl, _ = grpo_objective(
                                actor.evaluate_actions(inputs, actions),
                                old,
                                ref,
                                trajectory.advantage,
                                phase,
                            )
                            old_kl += (
                                float(kl.sum()) / len(trajectory.actions) / len(indices)
                            )
                if old_kl > phase.old_policy_target_kl:
                    stopped = True
                    break
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(indices), phase.batch_size):
                # Whole trajectories belong to a minibatch; causal windows are
                # chunked within each trajectory without changing its weight.
                for i in indices[offset : offset + phase.batch_size]:
                    trajectory = trajectories[i]
                    for inputs, actions, old, ref in trajectory.batches(
                        actor, phase.scoring_chunk_size
                    ):
                        values = grpo_objective(
                            actor.evaluate_actions(inputs, actions),
                            old,
                            ref,
                            trajectory.advantage,
                            phase,
                        )
                        (
                            values[0].sum() / len(trajectory.actions) / len(indices)
                        ).backward()
                        for key, value in zip(totals, values):
                            totals[key] += float(value.detach().sum()) / len(
                                trajectory.actions
                            )
                    scored += 1
            apply_gradients(
                actor, optimizer, scheduler, context.optimization["actor"].gradient_clip
            )
            optimizer_steps += 1
        if stopped:
            break
    if not scored:
        raise ValueError("GRPO old-policy KL stopped all optimizer transactions")
    buffer.consume()
    actor.grpo_policy_version = generation + 1
    return {
        **{key: value / scored for key, value in totals.items()},
        "optimizer_steps": optimizer_steps,
        "optimization_epochs": epoch + 1,
        "old_policy_kl_stopped": stopped,
    }
