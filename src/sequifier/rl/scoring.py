"""Bounded differentiable scores for causal action sequences."""

import torch

from sequifier.io.actions import action_batch
from sequifier.io.policy_window import PolicyWindowView
from sequifier.rl.window_storage import collated_windows


def sequence_log_probabilities(actor, episode, start, end, chunk_size, *, rounds=None):
    """One FP32 score per executable decision; context/padding never contribute."""
    actor.eval()
    view = PolicyWindowView(actor.config)
    chunks = (end - start + chunk_size - 1) // chunk_size
    for slot in range(chunks if rounds is None else rounds):
        offset = start + (slot if slot < chunks else 0) * chunk_size
        positions = range(offset, min(offset + chunk_size, end))
        with collated_windows(
            (view.episode_inputs(episode, t, None) for t in positions), actor.device
        ) as inputs:
            actions = action_batch(
                [episode.decision(t).action for t in positions],
                actor.config,
                device=actor.device,
                encoded=False,
            )
            scores = actor.evaluate_actions(inputs, actions).float()
            del inputs
            if not torch.isfinite(scores).all():
                raise ValueError("non-finite sequence action log likelihood")
            yield scores if slot < chunks else scores * 0


def sequence_score(actor, episode, start, end, chunk_size):
    return torch.stack(
        [
            scores.sum()
            for scores in sequence_log_probabilities(
                actor, episode, start, end, chunk_size
            )
        ]
    ).sum()
