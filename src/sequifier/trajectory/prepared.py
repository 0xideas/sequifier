"""Reconstruct split-owned decisions from overlapping prepared histories."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from sequifier.trajectory.sources import ResolvedSequenceSource


def prepared_table(source: ResolvedSequenceSource) -> pl.DataFrame:
    """Recover unique positions from existing windows, validating overlaps.

    Split target bounds select decisions. A preceding action may be recovered
    from stored context, but observations outside the split are not targets.
    """
    metadata = source.metadata
    if metadata is None:
        raise ValueError("prepared histories require metadata_config_path")
    from sequifier.config.depth_layout import depth_mask_metadata_key
    from sequifier.helpers import PANDAS_TO_TORCH_TYPES, read_data
    from sequifier.io.pt_payload import load_pt_payload, tabular_tensor_batch

    split_index = (
        metadata.named_split_index(source.declaration.split)
        if source.declaration.split is not None
        else None
    )

    identities = getattr(metadata, "sequence_identities", {})
    rows = {}
    for path in source.paths:
        if path.suffix.lower() == ".pt":
            batch = load_pt_payload(
                path, layouts=metadata.depth_layouts, n_classes=metadata.n_classes
            )
        else:
            window_length = (
                metadata.width_for_split(split_index)
                if split_index is not None
                else metadata.window_length
            )
            data = read_data(str(path), path.suffix.lower().lstrip("."))
            batch = tabular_tensor_batch(
                data,
                {
                    column: PANDAS_TO_TORCH_TYPES[dtype]
                    for column, dtype in metadata.column_data_types.items()
                },
                sorted(set(data["inputCol"].to_list())),
                window_length,
            )
        sequences, ids, starts, pads = (
            batch.sequences,
            batch.sequence_ids,
            batch.start_item_positions,
            batch.left_pad_lengths,
        )
        lows, highs = batch.split_start_item_positions, batch.split_end_item_positions
        for row in range(len(ids)):
            for offset in range(
                int(pads[row]), next(iter(sequences.values())).shape[1]
            ):
                position = int(starts[row]) + offset
                key = (int(ids[row]), position)
                values = {
                    column: tensor[row, offset].tolist()
                    for column, tensor in sequences.items()
                }
                values.update(
                    {
                        depth_mask_metadata_key(name): mask[row, offset].tolist()
                        for name, mask in batch.depth_valid_masks.items()
                    }
                )
                record = {
                    "sequenceId": key[0],
                    "itemPosition": position,
                    **values,
                    "__decision": int(lows[row]) <= position < int(highs[row]),
                }
                if identity := identities.get(str(key[0])):
                    for column, field in (
                        ("episodeId", "episode_id"),
                        ("__group", "group_id"),
                    ):
                        if identity.get(field) is not None:
                            record[column] = identity[field]
                if key in rows and rows[key] != record:
                    raise ValueError(
                        "prepared history windows disagree on overlapping positions or split bounds"
                    )
                rows[key] = record
    if not rows:
        raise ValueError("prepared source contains no valid positions")
    return pl.DataFrame(list(rows.values())).sort(["sequenceId", "itemPosition"])
