"""Trajectory-weighted clipped GRPO with a sampled frozen-reference penalty."""

import torch

from sequifier.rl.batching import MetricTotals, RolloutPartition, optimize_groups
from sequifier.rl.clipped_objective import clipped_policy_objective
from sequifier.rl.grpo_buffer import GRPOTrajectory, SealedGroupBuffer


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
            del inputs, actions, old, _


def update_grpo(session, buffer: SealedGroupBuffer | None):
    context = session.context
    actor, phase = context.actor.eval(), context.phase
    generation = session.transaction
    if buffer:
        buffer.require_fresh(generation, context.reference_identity)
    if actor.grpo_policy_version != generation:
        raise ValueError("GRPO collection generation differs from actor")
    with session.rounds.scope():
        if buffer:
            validate_old_scores(actor, buffer, phase.scoring_chunk_size)
    trajectories: tuple[GRPOTrajectory, ...] = buffer.trajectories if buffer else ()
    strategy = session.strategy
    partition = RolloutPartition(strategy, len(trajectories))
    totals = MetricTotals(
        "loss",
        "sampled_reference_penalty",
        "old_policy_approximate_kl",
        "clip_fraction",
    )
    scored = 0

    def chunks(group):
        for index in partition.local(group):
            trajectory = trajectories[index]
            for batch in trajectory.batches(actor, phase.scoring_chunk_size):
                yield (*batch, trajectory.advantage, len(trajectory.actions))

    def group_kl(group):
        total = 0.0
        with session.rounds.scope():
            for inputs, actions, old, ref, advantage, length in chunks(group):
                _, _, kl, _ = grpo_objective(
                    actor.evaluate_actions(inputs, actions), old, ref, advantage, phase
                )
                total += float(kl.sum()) / length
        return strategy.sum(total) / len(group)

    def backward(group, epoch):
        nonlocal scored
        for batch, active in partition.aligned(chunks(group)):
            inputs, actions, old, ref, advantage, length = batch
            values = grpo_objective(
                actor.evaluate_actions(inputs, actions), old, ref, advantage, phase
            )
            scale = strategy.world_size / len(group) / length if active else 0.0
            (values[0].sum() * scale).backward()
            if active:
                totals.add(dict(zip(totals, values)), length)
        scored += len(group)

    epochs, stopped, _ = optimize_groups(session, partition.total, group_kl, backward)
    if not scored:
        raise ValueError("GRPO old-policy KL stopped all optimizer transactions")
    if buffer:
        buffer.consume()
    actor.grpo_policy_version = generation + 1
    return {
        **{key: strategy.sum(total) / scored for key, total in totals.items()},
        "optimizer_steps": session.steps["actor"],
        "optimization_epochs": epochs,
        "old_policy_kl_stopped": stopped,
    }
