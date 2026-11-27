"""PPO objectives and sample-weighted independent actor/value optimization."""

from typing import Any

import torch

from sequifier.rl.batching import (
    MetricTotals,
    RolloutPartition,
    batches,
    optimize_groups,
)
from sequifier.rl.clipped_objective import clipped_policy_objective
from sequifier.rl.collection import stream_seed
from sequifier.rl.distributions import action_generator


def ppo_objectives(
    log_probability,
    old_log_probability,
    advantages,
    prediction,
    old_value,
    target,
    entropy,
    phase,
):
    policy = clipped_policy_objective(
        log_probability, old_log_probability, advantages, phase.clip_epsilon
    )
    policy_loss, ratio = policy.loss, policy.ratio
    value_error = (prediction.float() - target).square()
    if phase.value_clip_delta is not None:
        clipped_value = old_value + (prediction.float() - old_value).clamp(
            -phase.value_clip_delta, phase.value_clip_delta
        )
        value_error = torch.maximum(value_error, (clipped_value - target).square())
    value_loss = 0.5 * value_error
    approximate_kl = policy.approximate_kl
    tensors = (ratio, policy_loss, value_loss, entropy, approximate_kl)
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("non-finite PPO probability ratio or objective")
    return (
        policy_loss,
        value_loss,
        {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approximate_kl": approximate_kl,
            "clip_fraction": policy.clip_fraction,
            "ratio": ratio,
        },
    )


@torch.no_grad()
def validate_collection_scores(actor, value, buffer, batch_size):
    """Check both identity ratio and V before the first optimizer transaction."""
    actor.eval()
    value.eval()
    for start in range(0, len(buffer), batch_size):
        indices = range(start, min(start + batch_size, len(buffer)))
        with buffer.batch_scope(indices, actor.device) as batch:
            history, actions, old_logp, old_v, *_ = batch
            if not torch.allclose(
                actor.evaluate_actions(history, actions), old_logp, atol=1e-5, rtol=1e-4
            ):
                raise ValueError(
                    "PPO reconstructed action scores differ from collection"
                )
            if not torch.allclose(value(history), old_v, atol=1e-5, rtol=1e-4):
                raise ValueError("PPO reconstructed values differ from collection")
            del batch, history


def update_ppo(session, value, buffer):
    actor, phase = session.context.actor, session.context.phase
    policy_version = session.transaction
    strategy = session.strategy
    partition = RolloutPartition(strategy, len(buffer) if buffer else 0)
    if buffer:
        buffer.require_version(policy_version)
    if actor.ppo_policy_version != policy_version:
        raise ValueError(
            "PPO buffer has already been consumed or belongs to a stale actor"
        )
    actor.eval()  # Keep dropout off while retaining autograd.
    value.eval()
    with session.rounds.scope():
        if buffer:
            validate_collection_scores(actor, value, buffer, phase.batch_size)
    totals = MetricTotals(
        "policy_loss", "value_loss", "entropy", "approximate_kl", "clip_fraction"
    )
    ratio_chunks = []
    count = 0

    def chunks(group):
        for indices in batches(partition.local(group), phase.batch_size):
            with buffer.batch_scope(indices, actor.device) as batch:
                yield batch, indices

    def group_kl(group):
        total = 0.0
        with session.rounds.scope():
            for (history, actions, old_logp, _, _, _), _ in chunks(group):
                log_ratio = actor.evaluate_actions(history, actions) - old_logp
                kl = log_ratio.exp() - 1 - log_ratio
                if not torch.isfinite(kl).all():
                    raise ValueError("non-finite PPO KL")
                total += float(kl.sum())
        return strategy.sum(total) / len(group)

    def update_chunks(group):
        for batch, indices in chunks(group):
            yield (*batch, [partition.start + index for index in indices])

    def backward(group, epoch):
        nonlocal count
        stream = f"ppo-entropy-{policy_version}-{epoch}"
        for batch, active in partition.aligned(update_chunks(group)):
            history, actions, old_logp, old_v, advantages, target, indices = batch
            # Recorded decisions retain their entropy samples across partition sizes.
            generator = (
                [
                    action_generator(
                        actor.device,
                        stream_seed(
                            session.context.identity.sampling_seed, stream, index
                        ),
                    )
                    for index in indices
                ]
                if not actor.action_plan.entropy_is_exact
                and session.state.get("per_decision_entropy", not session.state)
                else None
            )
            _, scores, entropy = actor.action_scores(
                history, actions, entropy=True, generator=generator
            )
            policy_loss, value_loss, metrics = ppo_objectives(
                scores,
                old_logp,
                advantages,
                value(history),
                old_v,
                target,
                entropy,
                phase,
            )
            scale = strategy.world_size / len(group) if active else 0.0
            session.optimization["actor"].backward(
                (policy_loss - phase.entropy_coefficient * entropy).sum() * scale
            )
            session.optimization["value"].backward(
                phase.value_coefficient * value_loss.sum() * scale
            )
            if active:
                totals.add(metrics)
                ratio_chunks.append(metrics["ratio"].detach().cpu())
                count += len(old_v)

    epochs, stopped, norms = optimize_groups(
        session, partition.total, group_kl, backward
    )
    count = int(strategy.sum(count))
    if not count:
        raise ValueError(
            "PPO KL stopped all updates before an identity-ratio transaction"
        )
    ratios = torch.cat(
        [chunk for parts in strategy.gather_objects(ratio_chunks) for chunk in parts]
    )
    targets = torch.tensor(buffer.targets if buffer else (), dtype=torch.float32)
    predictions = []
    with torch.no_grad(), session.rounds.scope():
        for batch, _ in chunks(range(partition.total)):
            predictions.append(value(batch[0]).cpu())
    residuals = targets - torch.cat(predictions) if predictions else targets

    def variance(values):
        mean = strategy.sum(float(values.double().sum())) / partition.total
        return (
            strategy.sum(float((values.double() - mean).square().sum()))
            / partition.total
        )

    residual_variance, target_variance = variance(residuals), variance(targets)
    clocks = session.steps
    result: dict[str, Any] = {
        key: strategy.sum(total) / count for key, total in totals.items()
    }
    result.update(
        {
            "actor_optimizer_step": clocks["actor"],
            "value_optimizer_step": clocks["value"],
            "actor_grad_norm": norms["actor"],
            "value_grad_norm": norms["value"],
            "ratio_min": float(ratios.min()),
            "ratio_max": float(ratios.max()),
            "ratio_p01": float(torch.quantile(ratios, 0.01)),
            "ratio_p99": float(torch.quantile(ratios, 0.99)),
            "explained_variance": float(1 - residual_variance / target_variance)
            if target_variance > 1e-12
            else None,
            "optimization_epochs": epochs,
            "scored_decisions": count,
            "kl_stopped": stopped,
        }
    )
    actor.ppo_policy_version = policy_version + 1
    return result
