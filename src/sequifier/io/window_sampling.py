from collections.abc import Sequence
from typing import Optional

import torch
from torch import Tensor

from sequifier.config.depth_layout import depth_mask_metadata_key
from sequifier.helpers import WindowSampleIndex
from sequifier.io.batch import SequifierBatch
from sequifier.typechecking import beartype


def validate_split_bounds_available(
    start_item_positions: Optional[Tensor],
    split_start_item_positions: Optional[Tensor],
    split_end_item_positions: Optional[Tensor],
    source: str,
) -> None:
    """Reject stored windows without enforceable target bounds."""
    if any(
        value is None
        for value in (
            start_item_positions,
            split_start_item_positions,
            split_end_item_positions,
        )
    ):
        raise ValueError(
            f"Stored windows in {source!r} require start positions and split target "
            "bounds; re-run preprocessing with the current format."
        )


def target_valid_from_offsets(
    left_pad_lengths: Tensor,
    start_item_positions: Tensor,
    split_start_item_positions: Tensor,
) -> Tensor:
    """Return the first stored offset whose target belongs to this split."""
    relative_split_starts = split_start_item_positions - start_item_positions
    return torch.maximum(
        left_pad_lengths.to(dtype=torch.int64, device="cpu"),
        relative_split_starts.to(dtype=torch.int64, device="cpu"),
    )


@beartype
def build_window_batch(
    sequences: dict[str, Tensor],
    input_columns: Sequence[str],
    target_columns: Sequence[str],
    sample_index: WindowSampleIndex,
    logical_indices: Tensor | list[int],
    sample_is_real: Optional[Sequence[bool] | Tensor] = None,
    depth_valid_masks: Optional[dict[str, Tensor]] = None,
    start_item_positions: Optional[Tensor] = None,
    split_start_item_positions: Optional[Tensor] = None,
    split_end_item_positions: Optional[Tensor] = None,
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
    if any(
        value is None
        for value in (
            start_item_positions,
            split_start_item_positions,
            split_end_item_positions,
        )
    ):
        raise ValueError("Split-boundary masking requires all position tensors")
    assert start_item_positions is not None
    assert split_start_item_positions is not None
    assert split_end_item_positions is not None
    relative_positions = torch.arange(
        plan.resolved_view.view.context_length, dtype=torch.int64
    )
    target_item_positions = (
        start_item_positions[stored_rows, None]
        + input_starts[:, None]
        + plan.resolved_view.view.target_offset
        + relative_positions[None, :]
    )
    inside_split = (
        target_item_positions >= split_start_item_positions[stored_rows, None]
    ) & (target_item_positions < split_end_item_positions[stored_rows, None])
    metadata["target_valid_mask"] &= inside_split
    for name, mask in (depth_valid_masks or {}).items():
        metadata[depth_mask_metadata_key(name)] = plan.gather(
            mask, stored_rows, input_starts
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
