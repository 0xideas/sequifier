"""Independent seed streams and categorical online decision scoring."""

import hashlib
from dataclasses import dataclass

import torch

from sequifier.rl.distributions import CategoricalActionDistribution


def stream_seed(seed, stream, reset_count):
    value = f"sequifier-policy-v1:{seed}:{stream}:{reset_count}".encode()
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big") % (
        2**63
    )


@dataclass(frozen=True)
class PolicyDecision:
    domain_action: dict
    action: int
    log_probability: float
    entropy: float
    value: float


def validate_online_inputs(actor, inputs):
    inputs = inputs.to(actor.device)
    actor.model_inputs(inputs)
    return inputs


@torch.no_grad()
def decide(actor, value, inputs, generator, *, deterministic=False):
    """Score action and history-only V from exactly the same causal input."""
    actor.eval()
    if value is not None:
        value.eval()
    inputs = inputs.to(actor.device)
    features, metadata = actor.model_inputs(inputs)
    distribution = CategoricalActionDistribution(
        actor.prepared_logits(features, metadata)[:, -1]
    )
    action = distribution.select(deterministic, generator)
    estimate = (
        value.prepared_value(features, metadata)
        if value is not None
        else torch.zeros(1, device=actor.device)
    )
    if not torch.isfinite(estimate).all():
        raise ValueError("non-finite collection value")
    index = int(action[0])
    return PolicyDecision(
        {actor.config.action_column: actor.config.action_values[index]},
        index,
        float(distribution.log_prob(action)[0]),
        float(distribution.entropy()[0]),
        float(estimate[0]),
    )
