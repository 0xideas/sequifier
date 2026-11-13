"""One persistent environment slot; horizons cut history without resetting it."""

import torch

from sequifier.rl.collection import PolicyCursor, stream_seed
from sequifier.rl.ppo_buffer import PPOBufferBuilder, PPODecision


class PPOCollector:
    def __init__(self, actor, value, adapter, phase, seed):
        self.actor = actor
        self.value = value
        self.adapter = adapter
        self.phase = phase
        self.seed = seed
        self.cursor = PolicyCursor(
            actor,
            adapter,
            phase.environment.units,
            stream_seed(seed, "training-actions", 0),
        )

    @property
    def reset_count(self):
        return self.cursor.history.reset_count

    def reset(self):
        self.cursor.reset(
            self.adapter.reset(
                stream_seed(self.seed, "training-reset", self.reset_count)
            ),
            f"train-{self.reset_count}",
        )

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
        if self.cursor.ended:
            self.reset()
        builder = PPOBufferBuilder(
            self.actor.config,
            policy_version,
            horizon,
            self.cursor.history.frames,
            self.phase.max_buffer_bytes,
        )
        current_reference = len(builder.frames) - 1
        completed = []
        with torch.no_grad():
            for _ in range(horizon):
                if self.cursor.ended:
                    self.reset()
                    current_reference = builder.append_frame(
                        self.cursor.history.current
                    )
                if current_reference is None:
                    raise ValueError(
                        "live PPO decision lacks its observation reference"
                    )
                current = self.cursor.history.current
                decision = self.cursor.decide(self.value)
                result = self.cursor.step(decision)
                successor_reference = None
                successor_value = 0.0
                if result.observation is not None:
                    features, metadata = self.actor.model_inputs(self.cursor.inputs)
                    if not result.terminated:
                        successor_value = float(
                            self.value.prepared_value(features, metadata)[0]
                        )
                    if not torch.isfinite(
                        torch.tensor(successor_value, dtype=torch.float32)
                    ):
                        raise ValueError("non-finite PPO successor value")
                    successor_reference = builder.append_frame(
                        self.cursor.history.current
                    )
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
                        decision.domain_action,
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
                if self.cursor.ended:
                    completed.append(
                        {
                            "episode_id": current.episode_id,
                            "return": self.cursor.episode_return,
                            "length": self.cursor.episode_length,
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
            "history": self.cursor.history.state_dict(),
            "reset_count": self.reset_count,
            "generator": self.cursor.generator.get_state(),
            "ended": self.cursor.ended,
            "episode_return": self.cursor.episode_return,
            "episode_length": self.cursor.episode_length,
        }

    def load_state_dict(self, state, *, exact):
        reset_count = int(state["reset_count"])
        if reset_count < 1:
            raise ValueError("PPO checkpoint has an invalid reset count")
        self.cursor.generator.set_state(state["generator"])
        if exact:
            self.cursor.history.load_state_dict(state["history"])
            self.cursor.refresh()
            if (
                self.reset_count != reset_count
                or self.cursor.history.current.episode_id != f"train-{reset_count - 1}"
            ):
                raise ValueError(
                    "PPO checkpoint episode identity disagrees with reset count"
                )
            self.cursor.ended = bool(state["ended"])
            self.cursor.episode_return = float(state["episode_return"])
            self.cursor.episode_length = int(state["episode_length"])
        else:
            self.cursor.history.reset_count = reset_count
            self.cursor.ended = True
            self.cursor.episode_return = 0.0
            self.cursor.episode_length = 0
