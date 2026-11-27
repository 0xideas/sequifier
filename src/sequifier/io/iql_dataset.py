"""Fixed offline transitions with causal current and successor contexts."""

from dataclasses import dataclass

import torch

from sequifier.io.actions import action_batch
from sequifier.io.batch import SequifierInputs
from sequifier.io.policy_window import PolicyWindowView, checked_policy_features
from sequifier.io.tensors import TensorRecord
from sequifier.io.values import numeric_scalar


@dataclass(frozen=True)
class IQLBatch(TensorRecord):
    current: SequifierInputs
    successor: SequifierInputs
    action: dict[str, torch.Tensor]
    reward: torch.Tensor
    bootstrap: torch.Tensor


class IQLTransitionDataset:
    """Index eligible decisions while retaining the source's bounded episode cache."""

    required_columns = ("rewards", "terminated", "truncated")

    def __init__(self, decisions):
        self.decisions = decisions
        self.view = PolicyWindowView(decisions.config)
        self.cumulative_lengths = decisions.cumulative_lengths
        self.episode_ids = decisions.episode_ids
        for episode, length in zip(decisions.episodes, decisions.lengths, strict=True):
            transitions = episode.transitions
            missing = [
                field
                for field in self.required_columns
                if transitions is None or getattr(transitions, field) is None
            ]
            if missing:
                raise ValueError(
                    f"IQL episode {episode.episode_id!r} lacks required transition columns: {missing}"
                )
            assert transitions is not None
            for step in range(length):
                location = (
                    f"IQL episode {episode.episode_id!r} "
                    f"step {episode.first_position + step}"
                )
                reward = numeric_scalar(transitions.rewards[step], f"{location} reward")
                if not torch.isfinite(torch.tensor(reward, dtype=torch.float32)):
                    raise ValueError(f"{location}: reward exceeds finite float32 range")

    def validate_normalization(self, interface):
        for episode, length in zip(self.decisions.episodes, self.decisions.lengths):
            step = length - 1
            if self._ends_at_horizon(episode, step):
                continue
            inputs = self.view.successor_inputs(episode, step)
            checked_policy_features(inputs, interface)

    def _ends_at_horizon(self, episode, step):
        # A fragment beginning at zero with a previous action has no known offset.
        known_offset = episode.starts_episode or episode.first_position > 0
        horizon = self.view.config.episode_horizon
        return known_offset and episode.first_position + step + 1 == horizon

    def __len__(self):
        return len(self.decisions)

    def __getitem__(self, index):
        episode_index, step = self.decisions.locate(index)
        episode = self.decisions.episodes[episode_index]
        transitions = episode.transitions
        ends_at_horizon = self._ends_at_horizon(episode, step)
        current = self.view.episode_inputs(episode, step, None)
        next_inputs = self.view.successor_inputs(episode, step)
        return IQLBatch(
            current=current,
            successor=next_inputs,
            action=action_batch(
                [episode.actions[step]], self.decisions.config, encoded=False
            ),
            reward=torch.tensor([transitions.rewards[step]], dtype=torch.float32),
            bootstrap=torch.tensor(
                [not ends_at_horizon and not transitions.terminated[step]],
                dtype=torch.bool,
            ),
        )
