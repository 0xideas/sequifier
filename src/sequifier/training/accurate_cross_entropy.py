"""Hard-label cross-entropy that preserves small target-logit gradients."""

from __future__ import annotations

from torch import Tensor, nn

from sequifier.accurate_softmax import accurate_softmax_terms


class AccurateCrossEntropyLoss(nn.Module):
    """Per-example cross-entropy using a target-excluding log-sum-exp.

    The training loss service applies masking and dataset loss weights after this
    module returns its unreduced, optionally class-weighted losses.
    """

    def __init__(self, weight: Tensor | None = None, reduction: str = "none") -> None:
        super().__init__()
        if reduction != "none":
            raise ValueError("AccurateCrossEntropyLoss requires reduction='none'.")
        self.register_buffer("weight", weight)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if logits.ndim != 2 or target.ndim != 1 or logits.shape[0] != target.shape[0]:
            raise ValueError(
                "Expected logits of shape (N, C) and targets of shape (N,)."
            )
        if logits.shape[1] < 1:
            raise ValueError("AccurateCrossEntropyLoss requires at least one class.")

        # The target slot must be excluded from the sum, not set to zero.
        # In particular, its gradient comes from all the non-target differences
        # and remains the negative sum of their gradients at large margins.
        _, log_normalizer = accurate_softmax_terms(logits, target[:, None])
        loss = log_normalizer.squeeze(-1)
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss
