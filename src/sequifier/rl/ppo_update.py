"""PPO objectives and sample-weighted independent actor/value optimization."""

from typing import Any

import torch

from sequifier.rl.distributions import CategoricalActionDistribution


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
    log_ratio = log_probability.float() - old_log_probability.float()
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1 - phase.clip_epsilon, 1 + phase.clip_epsilon)
    policy_loss = -torch.minimum(ratio * advantages, clipped * advantages)
    value_error = (prediction.float() - target).square()
    if phase.value_clip is not None:
        clipped_value = old_value + (prediction.float() - old_value).clamp(
            -phase.value_clip, phase.value_clip
        )
        value_error = torch.maximum(value_error, (clipped_value - target).square())
    value_loss = 0.5 * value_error
    approximate_kl = ratio - 1 - log_ratio
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
            "clip_fraction": ((ratio - 1).abs() > phase.clip_epsilon).float(),
            "ratio": ratio,
        },
    )


@torch.no_grad()
def validate_collection_scores(actor, value, buffer, batch_size):
    """Check both identity ratio and V before the first optimizer transaction."""
    actor.eval()
    value.eval()
    for start in range(0, len(buffer), batch_size):
        history, actions, old_logp, old_v, _, _ = buffer.batch(
            range(start, min(start + batch_size, len(buffer))), actor.device
        )
        if not torch.allclose(
            actor.evaluate_actions(history, actions), old_logp, atol=1e-5, rtol=1e-4
        ):
            raise ValueError("PPO reconstructed action scores differ from collection")
        if not torch.allclose(value(history), old_v, atol=1e-5, rtol=1e-4):
            raise ValueError("PPO reconstructed values differ from collection")


def update_ppo(session, value, buffer):
    actor, phase = session.context.actor, session.context.phase
    policy_version, clocks = session.transaction, session.steps
    buffer.require_version(policy_version)
    if actor.ppo_policy_version != policy_version:
        raise ValueError(
            "PPO buffer has already been consumed or belongs to a stale actor"
        )
    actor.eval()  # Keep dropout off while retaining autograd.
    value.eval()
    validate_collection_scores(actor, value, buffer, phase.minibatch_size)
    totals = {
        key: 0.0
        for key in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approximate_kl",
            "clip_fraction",
        )
    }
    ratio_chunks = []
    count = 0
    norms = {"actor": [], "value": []}
    stopped = False
    group_size = phase.minibatch_size * phase.accumulation_steps
    epochs = 0
    for epoch in range(phase.optimization_epochs):
        order = torch.randperm(len(buffer), generator=session.sampler).tolist()
        for start in range(0, len(order), group_size):
            group = order[start : start + group_size]
            minibatches = [
                group[offset : offset + phase.minibatch_size]
                for offset in range(0, len(group), phase.minibatch_size)
            ]
            if phase.target_kl is not None:
                kl_sum = 0.0
                with torch.no_grad():
                    for indices in minibatches:
                        history, actions, old_logp, _, _, _ = buffer.batch(
                            indices, actor.device
                        )
                        log_ratio = actor.evaluate_actions(history, actions) - old_logp
                        kl = log_ratio.exp() - 1 - log_ratio
                        if not torch.isfinite(kl).all():
                            raise ValueError("non-finite PPO KL")
                        kl_sum += float(kl.sum())
                if kl_sum / len(group) > phase.target_kl:
                    stopped = True
                    break
            session.zero_grad()
            for indices in minibatches:
                history, actions, old_logp, old_v, advantages, target = buffer.batch(
                    indices, actor.device
                )
                distribution = CategoricalActionDistribution(actor(history)[:, -1])
                policy_loss, value_loss, metrics = ppo_objectives(
                    distribution.log_prob(actions),
                    old_logp,
                    advantages,
                    value(history),
                    old_v,
                    target,
                    distribution.entropy(),
                    phase,
                )
                actor_loss = (
                    policy_loss - phase.entropy_coefficient * metrics["entropy"]
                ).sum() / len(group)
                critic_loss = phase.value_coefficient * value_loss.sum() / len(group)
                if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
                    raise ValueError("non-finite weighted PPO loss")
                actor_loss.backward()
                critic_loss.backward()
                for key in totals:
                    totals[key] += float(metrics[key].detach().sum())
                ratio_chunks.append(metrics["ratio"].detach().cpu())
                count += len(indices)
            for role, norm in session.step().items():
                norms[role].append(norm)
        epochs = epoch + 1
        if stopped:
            break
    if not count:
        raise ValueError(
            "PPO KL stopped all updates before an identity-ratio transaction"
        )
    ratios = torch.cat(ratio_chunks)
    targets = torch.tensor(buffer.targets, dtype=torch.float32)
    with torch.no_grad():
        predictions = []
        for start in range(0, len(buffer), phase.minibatch_size):
            history, *_ = buffer.batch(
                range(start, min(start + phase.minibatch_size, len(buffer))),
                actor.device,
            )
            predictions.append(value(history).cpu())
        residual_variance = (targets - torch.cat(predictions)).var(unbiased=False)
        target_variance = targets.var(unbiased=False)
    result: dict[str, Any] = {key: total / count for key, total in totals.items()}
    result.update(
        {
            "actor_optimizer_step": clocks["actor"],
            "value_optimizer_step": clocks["value"],
            "actor_grad_norm": sum(norms["actor"]) / len(norms["actor"]),
            "value_grad_norm": sum(norms["value"]) / len(norms["value"]),
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
