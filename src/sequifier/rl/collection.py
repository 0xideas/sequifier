"""Independent seed streams and categorical online decision scoring."""

import hashlib
import math
from dataclasses import dataclass

import torch

from sequifier.io.actions import action_record, decode_action
from sequifier.rl.distributions import action_generator
from sequifier.rl.environment import validate_reset, validate_step
from sequifier.rl.history import PolicyHistory
from sequifier.trajectory.returns import remaining_return


def stream_seed(seed, stream, reset_count):
    value = f"sequifier-policy-v1:{seed}:{stream}:{reset_count}".encode()
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big") % (
        2**63
    )


@dataclass(frozen=True)
class PolicyDecision:
    domain_action: dict
    action: int | dict
    log_probability: float
    entropy: float
    value: float


class PolicyCursor:
    """Own causal history and action RNG for one live environment.

    Decision and execution are separate so collectors can score a reference and
    account for issued actions before handing control to the environment.
    """

    def __init__(self, actor, adapter, units, seed, *, strict_vocabulary=True):
        self.actor, self.adapter, self.units = actor, adapter, units
        self.history = PolicyHistory(actor.config, strict_vocabulary=strict_vocabulary)
        self.generator = action_generator(actor.device, seed)
        self.desired_return = None
        self.episode_return, self.episode_length, self.ended = 0.0, 0, True
        self.last_result = None

    def refresh(self):
        self.inputs = self.history.inputs().to(self.actor.device)
        self.actor.model_inputs(self.inputs)

    def reset(self, result, episode_id, desired_return=None):
        result = validate_reset(result, self.units)
        if self.actor.config.return_conditioned and desired_return is None:
            desired_return = self.actor.config.initial_desired_return
            if desired_return is None:
                raise ValueError(
                    "return-conditioned collection needs an initial desired return"
                )
        self.history.reset(
            result.observation, result.static_inputs, episode_id, desired_return
        )
        self.desired_return = desired_return
        self.episode_return, self.episode_length, self.ended = 0.0, 0, False
        self.last_result = None
        self.refresh()

    @torch.no_grad()
    def decide(self, value=None, *, deterministic=False):
        """Score action and history-only V from exactly the same causal input."""
        actor = self.actor
        actor.eval()
        if value is not None:
            value.eval()
        distribution = actor.distribution(self.inputs)
        features, metadata = distribution.features, distribution.metadata
        action = distribution.select(deterministic, self.generator)
        estimate = (
            value.prepared_value(features, metadata)
            if value is not None
            else torch.zeros(1, device=actor.device)
        )
        if not torch.isfinite(estimate).all():
            raise ValueError("non-finite collection value")
        index = action_record(action)
        return PolicyDecision(
            decode_action(index, actor.config),
            index,
            float(distribution.log_prob(action)[0]),
            float(distribution.entropy(self.generator)[0]),
            float(estimate[0]),
        )

    def step(self, decision):
        result = validate_step(self.adapter.step(decision.domain_action), self.units)
        self.last_result = result
        self.episode_return += float(result.reward)
        self.episode_length += 1
        self.ended = result.terminated or result.truncated
        if not math.isfinite(self.episode_return):
            raise ValueError("non-finite collected episode return")
        if self.actor.config.return_conditioned:
            assert self.desired_return is not None
            self.desired_return = remaining_return(
                self.desired_return, result.reward, self.actor.config.returns.gamma
            )
        if result.observation is not None:
            self.history.advance(
                decision.action, result.observation, self.desired_return
            )
            self.refresh()
        return result
