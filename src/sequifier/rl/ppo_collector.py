"""One persistent environment slot; horizons cut history without resetting it."""

import math

import torch

from sequifier.rl.collection import decide, stream_seed, validate_online_inputs
from sequifier.rl.distributions import action_generator
from sequifier.rl.environment import validate_reset, validate_step
from sequifier.rl.history import PolicyHistory
from sequifier.rl.ppo_buffer import PPOBufferBuilder, PPODecision


class PPOCollector:
    def __init__(self, actor, value, adapter, phase, seed):
        self.actor = actor
        self.value = value
        self.adapter = adapter
        self.phase = phase
        self.seed = seed
        self.history = PolicyHistory(actor.config, strict_vocabulary=True)
        self.generator = action_generator(
            actor.device, stream_seed(seed, "training-actions", 0)
        )
        self.ended = True
        self.episode_return = 0.0
        self.episode_length = 0

    @property
    def reset_count(self):
        return self.history.reset_count

    def reset(self):
        result = validate_reset(
            self.adapter.reset(
                stream_seed(self.seed, "training-reset", self.reset_count)
            ),
            self.phase.environment.units,
        )
        self.history.reset(
            result.observation, result.static_inputs, f"train-{self.reset_count}"
        )
        self.ended = False
        self.episode_return = 0.0
        self.episode_length = 0
        validate_online_inputs(self.actor, self.history.inputs())

    def collect(self, horizon, policy_version):
        if self.actor.ppo_policy_version != policy_version:
            raise ValueError("collection policy generation disagrees with the actor")
        self.actor.eval()
        self.value.eval()
        # Preflight before the first reset where possible; exact domain size is
        # checked again after reset and on each insertion.
        minimum = (2 * horizon + self.actor.config.context_length) * (
            1024 + 128 * len(self.actor.interface.input_columns)
        ) + horizon * 1024
        if minimum > self.phase.max_buffer_bytes:
            raise ValueError(
                f"PPO buffer requires approximately {minimum} bytes before collection"
            )
        if self.ended:
            self.reset()
        builder = PPOBufferBuilder(
            self.actor.config,
            policy_version,
            horizon,
            self.history.frames,
            self.phase.max_buffer_bytes,
        )
        current_reference = len(builder.frames) - 1
        completed = []
        with torch.no_grad():
            for _ in range(horizon):
                if self.ended:
                    self.reset()
                    current_reference = builder.append_frame(self.history.current)
                if current_reference is None:
                    raise ValueError(
                        "live PPO decision lacks its observation reference"
                    )
                current = self.history.current
                decision = decide(
                    self.actor, self.value, self.history.inputs(), self.generator
                )
                result = validate_step(
                    self.adapter.step(decision.domain_action),
                    self.phase.environment.units,
                )
                successor_reference = None
                successor_value = 0.0
                if result.observation is not None:
                    self.history.advance(decision.action, result.observation)
                    successor_inputs = validate_online_inputs(
                        self.actor, self.history.inputs()
                    )
                    if not result.terminated:
                        successor_value = float(self.value(successor_inputs)[0])
                    if not torch.isfinite(
                        torch.tensor(successor_value, dtype=torch.float32)
                    ):
                        raise ValueError("non-finite PPO successor value")
                    successor_reference = builder.append_frame(self.history.current)
                reward = float(result.reward) * self.phase.reward_scale
                if not torch.isfinite(
                    torch.tensor([reward, successor_value], dtype=torch.float32)
                ).all():
                    raise ValueError("reward or successor V exceeds finite FP32 range")
                builder.append(
                    PPODecision(
                        current_reference,
                        successor_reference,
                        decision.action,
                        self.actor.config.action_values[decision.action],
                        decision.log_probability,
                        decision.value,
                        decision.entropy,
                        reward,
                        float(result.reward),
                        successor_value,
                        result.terminated,
                        result.truncated,
                        not result.terminated,
                        current.episode_id,
                        current.step_id,
                        policy_version,
                    )
                )
                current_reference = successor_reference
                self.episode_return += float(result.reward)
                if not math.isfinite(self.episode_return):
                    raise ValueError("non-finite training episode return")
                self.episode_length += 1
                self.ended = result.terminated or result.truncated
                if self.ended:
                    completed.append(
                        {
                            "episode_id": current.episode_id,
                            "return": self.episode_return,
                            "length": self.episode_length,
                            "terminated": result.terminated,
                            "truncated": result.truncated,
                            "success": result.success,
                        }
                    )
        return builder.seal(
            self.phase.gamma, self.phase.gae_lambda, self.phase.normalize_advantages
        ), completed

    def state_dict(self):
        return {
            "history": self.history.state_dict(),
            "reset_count": self.reset_count,
            "generator": self.generator.get_state(),
            "ended": self.ended,
            "episode_return": self.episode_return,
            "episode_length": self.episode_length,
        }

    def load_state_dict(self, state, *, exact):
        reset_count = int(state["reset_count"])
        if reset_count < 1:
            raise ValueError("PPO checkpoint has an invalid reset count")
        self.generator.set_state(state["generator"])
        if exact:
            self.history.load_state_dict(state["history"])
            if (
                self.reset_count != reset_count
                or self.history.current.episode_id != f"train-{reset_count - 1}"
            ):
                raise ValueError(
                    "PPO checkpoint episode identity disagrees with reset count"
                )
            self.ended = bool(state["ended"])
            self.episode_return = float(state["episode_return"])
            self.episode_length = int(state["episode_length"])
        else:
            self.history.reset_count = reset_count
            self.ended = True
            self.episode_return = 0.0
            self.episode_length = 0
