"""Bounded differentiable scores for causal action sequences."""

from typing import cast

import torch

from sequifier.io.actions import action_batch
from sequifier.io.batch import SequifierInputs, collate_model_inputs
from sequifier.io.policy_window import PolicyWindowView


def sequence_log_probabilities(actor, episode, start, end, chunk_size):
    """One FP32 score per executable decision; context/padding never contribute."""
    actor.eval()
    view = PolicyWindowView(actor.config)
    for offset in range(start, end, chunk_size):
        positions = range(offset, min(offset + chunk_size, end))
        inputs = cast(
            SequifierInputs,
            collate_model_inputs(
                [view.episode_inputs(episode, t, None) for t in positions]
            ),
        ).to(actor.device)
        actions = action_batch(
            [episode.actions[t] for t in positions],
            actor.config,
            device=actor.device,
            encoded=False,
        )
        scores = actor.evaluate_actions(inputs, actions).float()
        if not torch.isfinite(scores).all():
            raise ValueError("non-finite sequence action log likelihood")
        yield scores


def sequence_score(actor, episode, start, end, chunk_size):
    return torch.stack(
        [
            scores.sum()
            for scores in sequence_log_probabilities(
                actor, episode, start, end, chunk_size
            )
        ]
    ).sum()
