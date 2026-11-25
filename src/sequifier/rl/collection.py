"""Independent seed streams and categorical online decision scoring."""

import hashlib
import math
from dataclasses import dataclass, replace

import torch

from sequifier.contract_versions import ContractVersions
from sequifier.io.actions import action_record
from sequifier.rl.distributions import action_generator
from sequifier.rl.environment import validate_reset, validate_step
from sequifier.rl.history import PolicyHistory
from sequifier.rl.window_storage import tensor_storage_bytes
from sequifier.runtime.resource_accounting import allocation
from sequifier.trajectory.returns import remaining_return


def stream_seed(seed, stream, reset_count):
    value = f"sequifier-policy-v{ContractVersions.POLICY_COLLECTION_SEED}:{seed}:{stream}:{reset_count}".encode()
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

    def __init__(
        self,
        actor,
        adapter,
        units,
        seed,
        *,
        strict_vocabulary=True,
        defer_device_transfer=False,
    ):
        self.actor, self.adapter, self.units = actor, adapter, units
        self.defer_device_transfer = defer_device_transfer
        self.history = PolicyHistory(
            actor.config.model_copy(update={"input_encoding": "raw"}),
            strict_vocabulary=strict_vocabulary,
        )
        self.generator = action_generator(actor.device, seed)
        self.desired_return = None
        self.episode_return, self.episode_length, self.ended = 0.0, 0, True
        self.last_result = None
        self._inputs = None
        self._window_storage = allocation("reconstructed_windows")

    @property
    def inputs(self):
        if self._inputs is None:
            raise ValueError("policy cursor has no reconstructed inputs")
        return self._inputs

    def refresh(self):
        inputs = self.history.inputs()
        moved = inputs if self.defer_device_transfer else inputs.to(self.actor.device)
        if self._window_storage.accounting is not None:
            seen = set()
            size = tensor_storage_bytes(moved, seen)
            self._window_storage.resize(
                size
                + tensor_storage_bytes(inputs, seen)
                + tensor_storage_bytes(self._inputs, seen)
            )
            self._window_storage.resize(size)
        self._inputs = moved

    def close(self):
        self._inputs = None
        self.last_result = None
        self.history.frames.clear()
        self._window_storage.close()

    def reset(
        self,
        result,
        episode_id,
        desired_return=None,
        *,
        earlier=(),
        observation_masks=None,
        previous_action=None,
    ):
        result = validate_reset(result, self.units, compact_images=True)
        if self.actor.config.return_conditioned and desired_return is None:
            desired_return = self.actor.config.initial_desired_return
            if desired_return is None:
                raise ValueError(
                    "return-conditioned collection needs an initial desired return"
                )
        self.history.reset(
            result.observation,
            result.static_inputs,
            episode_id,
            desired_return,
            earlier=earlier,
            observation_masks=observation_masks,
            previous_action=previous_action,
        )
        self.desired_return = desired_return
        self.episode_return, self.episode_length, self.ended = 0.0, 0, False
        self.last_result = None
        self.refresh()

    @torch.no_grad()
    def decide(self, value=None, *, deterministic=False, compute_entropy=True):
        """Score action and history-only V from exactly the same causal input."""
        actor = self.actor
        actor.eval()
        if value is not None:
            value.eval()
        record, score, entropy = actor.action_scores(
            self.inputs,
            deterministic=deterministic,
            generator=self.generator,
            entropy=compute_entropy,
        )
        estimate = (
            value(self.inputs)
            if value is not None
            else torch.zeros(1, device=actor.device)
        )
        if not torch.isfinite(estimate).all():
            raise ValueError("non-finite collection value")
        decision = actor.action_plan.from_record(record).cpu()
        return PolicyDecision(
            actor.action_plan.domain_action(decision),
            action_record(actor.action_plan.to_record(decision)),
            float(score[0]),
            float(entropy[0]) if compute_entropy else 0.0,
            float(estimate[0]),
        )

    def step(self, decision):
        return self.accept_step(decision, self.adapter.step(decision.domain_action))

    def accept_step(self, decision, result):
        """Validate and commit exactly one simulator response."""
        result = validate_step(result, self.units, compact_images=True)
        self.episode_return += float(result.reward)
        self.episode_length += 1
        if (
            self.episode_length == self.actor.config.episode_horizon
            and not result.terminated
        ):
            # The finite return ends here even when the simulator could continue.
            result = replace(result, terminated=True, truncated=False)
        self.last_result = result
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
