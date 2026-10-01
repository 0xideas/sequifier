from collections.abc import Sequence
from typing import Optional

import torch
from torch import Tensor

from sequifier.config.depth_layout import depth_mask_metadata_key
from sequifier.helpers import WindowSampleIndex
from sequifier.io.batch import SequifierBatch
from sequifier.typechecking import beartype


def requires_split_bounds(config: object) -> bool:
    """Return whether resolved dataset metadata declares preceding context."""
    part = getattr(config, "part", None)
    metadata = getattr(part, "metadata", None)
    split_context = getattr(metadata, "split_context", None)
    return getattr(split_context, "mode", "isolated") == "preceding"


def validate_split_bounds_available(
    config: object,
    start_item_positions: Optional[Tensor],
    split_start_item_positions: Optional[Tensor],
    split_end_item_positions: Optional[Tensor],
    source: str,
) -> None:
    """Reject a preceding-context contract without enforceable target bounds."""
    if requires_split_bounds(config) and any(
        value is None
        for value in (
            start_item_positions,
            split_start_item_positions,
            split_end_item_positions,
        )
    ):
        raise ValueError(
            f"Preceding split context in {source!r} requires stored split target "
            "bounds; re-run preprocessing rather than risking target leakage."
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
    split_bounds_present = (
        start_item_positions is not None
        and split_start_item_positions is not None
        and split_end_item_positions is not None
    )
    if split_bounds_present:
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
    elif any(
        value is not None
        for value in (
            start_item_positions,
            split_start_item_positions,
            split_end_item_positions,
        )
    ):
        raise ValueError("Split-boundary masking requires all position tensors")
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
