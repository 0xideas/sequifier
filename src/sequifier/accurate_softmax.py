"""Shared numerical core for accurate softmax and hard-label cross-entropy."""

import torch
import torch.nn.functional as F
from torch import Tensor


def accurate_softmax_terms(
    scores: Tensor, selected_index: Tensor
) -> tuple[Tensor, Tensor]:
    """Return shifted scores and log normalizer with one score factored out.

    ``selected_index`` has one index per row and a final dimension of size one.
    The selected shifted score is exactly zero, so its gradient comes from the
    differences to the other scores rather than a rounded probability.
    """
    if scores.shape[-1] == 1:
        zero = scores - scores
        return zero, zero

    selected_mask = (
        torch.arange(scores.shape[-1], device=scores.device) == selected_index
    )
    selected = torch.where(selected_mask, scores, 0.0).sum(dim=-1, keepdim=True)
    shifted = torch.where(selected_mask, 0.0, scores - selected)
    others = torch.where(selected_mask, -torch.inf, shifted)
    log_normalizer = F.softplus(torch.logsumexp(others, dim=-1, keepdim=True))
    return shifted, log_normalizer


def accurate_softmax(scores: Tensor) -> Tensor:
    """Max-excluding softmax from arXiv:2610.31642, Equation 5."""
    working_scores = (
        scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
    )
    max_index = working_scores.argmax(dim=-1, keepdim=True)
    shifted, log_normalizer = accurate_softmax_terms(working_scores, max_index)
    return torch.exp(shifted - log_normalizer).to(dtype=scores.dtype)
