from collections.abc import Sequence
from typing import Optional

import torch
from torch import Tensor

from sequifier.helpers import WindowSampleIndex
from sequifier.io.batch import SequifierBatch


def build_window_batch(
    sequences: dict[str, Tensor],
    input_columns: Sequence[str],
    target_columns: Sequence[str],
    sample_index: WindowSampleIndex,
    logical_indices: Tensor | list[int],
    sample_is_real: Optional[Sequence[bool] | Tensor] = None,
) -> SequifierBatch:
    """Gather one batch of virtual model windows from stored tensors."""
    stored_rows, input_starts = sample_index.resolve(logical_indices)
    plan = sample_index.plan

    inputs = {
        column: plan.gather(
            sequences[column],
            stored_rows,
            input_starts,
        )
        for column in input_columns
    }
    targets = {
        column: plan.gather(
            sequences[column],
            stored_rows,
            input_starts,
            target=True,
        )
        for column in target_columns
    }
    metadata = plan.build_masks(
        sample_index.left_pad_lengths[stored_rows],
        input_starts,
    )
    if sample_is_real is not None:
        metadata["sample_valid_mask"] = torch.as_tensor(
            sample_is_real,
            dtype=torch.bool,
        )

    return SequifierBatch(
        inputs=inputs,
        targets=targets,
        metadata=metadata,
    )
