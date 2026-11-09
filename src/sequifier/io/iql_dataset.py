"""Fixed offline transitions with causal current and successor contexts."""

import hashlib
import json
from bisect import bisect_right
from dataclasses import dataclass

import torch

from sequifier.io.batch import SequifierInputs
from sequifier.io.policy_window import PolicyWindowView, action_index
from sequifier.io.tensors import TensorRecord
from sequifier.io.values import numeric_scalar


@dataclass(frozen=True)
class IQLBatch(TensorRecord):
    current: SequifierInputs
    successor: SequifierInputs
    action: torch.Tensor
    reward: torch.Tensor
    bootstrap: torch.Tensor


class IQLTransitionDataset:
    """Index eligible decisions while retaining the source's bounded episode cache."""

    def __init__(self, policy_dataset):
        if policy_dataset.config.return_conditioned:
            raise ValueError("IQL requires an unconditioned actor")
        self.policy_dataset = policy_dataset
        self.view = PolicyWindowView(policy_dataset.config)
        self.cumulative_lengths = policy_dataset.cumulative_lengths
        self.episode_ids = policy_dataset.episode_ids
        digest = hashlib.sha256()
        for episode_index, episode in enumerate(policy_dataset.episodes):
            source = episode_index
            transitions = episode.transitions
            if transitions is None or any(
                getattr(transitions, field) is None
                for field in ("rewards", "terminated", "truncated", "bootstrap_allowed")
            ):
                raise ValueError(
                    f"IQL source {source!r} episode {episode.episode_id!r} "
                    "needs complete transition columns"
                )
            for step in range(len(episode.actions)):
                location = (
                    f"IQL source {source!r} episode {episode.episode_id!r} "
                    f"step {episode.item_positions[step]}"
                )
                reward = numeric_scalar(transitions.rewards[step], f"{location} reward")
                if not torch.isfinite(torch.tensor(reward, dtype=torch.float32)):
                    raise ValueError(f"{location}: reward exceeds finite float32 range")
                terminated = transitions.terminated[step]
                bootstrap = transitions.bootstrap_allowed[step]
                successor = self._successor(episode, step)
                if terminated and bootstrap:
                    raise ValueError(f"{location}: termination forbids bootstrap")
                if not terminated and not bootstrap:
                    raise ValueError(
                        f"{location}: nonterminal transition must bootstrap"
                    )
                if bootstrap and successor is None:
                    raise ValueError(f"{location}: bootstrap needs the true successor")
                if successor is not None and set(successor) != set(
                    episode.observations
                ):
                    raise ValueError(
                        f"{location}: successor fields differ from observations"
                    )
                if successor is not None and step + 1 < len(episode.actions):
                    if successor != episode.observation(step + 1):
                        raise ValueError(
                            f"{location}: successor breaks episode continuity"
                        )
            digest.update(
                json.dumps(episode.model_dump(mode="json"), sort_keys=True).encode()
            )
        self.data_contract = digest.hexdigest()

    def use_config(self, config):
        self.view = PolicyWindowView(config)

    def validate_normalization(self, actor):
        # Interior successors are ordinary observations, already checked by the
        # policy dataset. Only the final successor can contain additional values.
        for episode in self.policy_dataset.episodes:
            step = len(episode.actions) - 1
            successor = self._successor(episode, step)
            if successor is None:
                continue
            inputs = self.view.successor_inputs(episode, step, successor)
            features, _ = actor.model_inputs(inputs.to(actor.device))
            if any(
                t.is_floating_point() and not torch.isfinite(t).all()
                for t in features.values()
            ):
                raise ValueError(
                    f"episode {episode.episode_id}: non-finite normalized IQL successor"
                )

    @staticmethod
    def _successor(episode, step):
        transitions = episode.transitions
        if transitions is not None and transitions.successors is not None:
            supplied = transitions.successors[step]
            if supplied is not None:
                return supplied
        if step + 1 < len(episode.actions):
            return episode.observation(step + 1)
        return None

    def __len__(self):
        return len(self.policy_dataset)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect_right(self.cumulative_lengths, index)
        step = index - (
            self.cumulative_lengths[episode_index - 1] if episode_index else 0
        )
        episode = self.policy_dataset.episodes[episode_index]
        transitions = episode.transitions
        successor = self._successor(episode, step)
        current = self.view.episode_inputs(episode, step, None)
        next_inputs = self.view.successor_inputs(
            episode,
            step,
            successor if successor is not None else episode.observation(step),
        )
        return IQLBatch(
            current=current,
            successor=next_inputs,
            action=torch.tensor(
                [action_index(episode.actions[step], self.policy_dataset.config)],
                dtype=torch.long,
            ),
            reward=torch.tensor([transitions.rewards[step]], dtype=torch.float32),
            bootstrap=torch.tensor(
                [transitions.bootstrap_allowed[step]], dtype=torch.bool
            ),
        )
