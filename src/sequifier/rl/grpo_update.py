"""Trajectory-weighted clipped GRPO with a sampled frozen-reference penalty."""

import torch

from sequifier.rl.batching import MetricTotals, shuffled_groups
from sequifier.rl.clipped_objective import clipped_policy_objective


def grpo_objective(
    log_probability, old_log_probability, reference_log_probability, advantage, phase
):
    policy = clipped_policy_objective(
        log_probability, old_log_probability, advantage, phase.clip_epsilon
    )
    reference_delta = reference_log_probability.float() - log_probability.float()
    penalty = torch.expm1(reference_delta) - reference_delta
    loss = policy.loss + phase.reference_kl_coefficient * penalty
    old_kl = policy.approximate_kl
    if any(not torch.isfinite(t).all() for t in (policy.ratio, penalty, loss, old_kl)):
        raise ValueError(
            "non-finite GRPO ratio, sampled reference penalty or objective"
        )
    return loss, penalty, old_kl, policy.clip_fraction


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


def update_grpo(session, buffer):
    context = session.context
    actor, phase = context.actor.eval(), context.phase
    generation = session.transaction
    buffer.require_fresh(generation, context.reference_identity)
    if actor.grpo_policy_version != generation:
        raise ValueError("GRPO collection generation differs from actor")
    validate_old_scores(actor, buffer, phase.scoring_chunk_size)
    trajectories = buffer.trajectories
    totals = MetricTotals(
        "loss",
        "sampled_reference_penalty",
        "old_policy_approximate_kl",
        "clip_fraction",
    )
    scored, stopped = 0, False
    size = phase.batch_size * phase.accumulation_batches
    epoch = -1
    for epoch in range(phase.optimization_epochs):
        for indices in shuffled_groups(len(trajectories), size, session.sampler):
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
            session.zero_grad()
            for i in indices:
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
                    totals.add(dict(zip(totals, values)), len(trajectory.actions))
                scored += 1
            session.step()
        if stopped:
            break
    if not scored:
        raise ValueError("GRPO old-policy KL stopped all optimizer transactions")
    buffer.consume()
    actor.grpo_policy_version = generation + 1
    return {
        **totals.means(scored),
        "optimizer_steps": session.steps["actor"],
        "optimization_epochs": epoch + 1,
        "old_policy_kl_stopped": stopped,
    }
