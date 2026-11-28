"""Reconstruct split-owned decisions from overlapping prepared histories."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from sequifier.trajectory.sources import ResolvedSequenceSource


def prepared_table(
    source: ResolvedSequenceSource,
) -> tuple[pl.DataFrame, dict[int, tuple[int, int]]]:
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
        else metadata.split_index
    )

    identities = getattr(metadata, "sequence_identities", {})
    rows = {}
    bounds = {}
    for path in source.paths:
        if path.suffix.lower() == ".pt":
            batch = load_pt_payload(
                path, layouts=metadata.depth_layouts, n_classes=metadata.n_classes
            )
        else:
            data = read_data(str(path), path.suffix.lower().lstrip("."))
            if "inputCol" not in data.columns:
                raise ValueError(
                    f"{path}: expected prepared sequence windows with inputCol. "
                    "For fresh inference rows, select input.layout: raw and supply "
                    "input.metadata_config_path."
                )
            steps = {name for name in data.columns if name.isdigit()}
            window_length = len(steps)
            allowed = (
                [metadata.width_for_split(split_index)]
                if split_index is not None
                else metadata.window_lengths or [metadata.window_length]
            )
            if window_length not in allowed or steps != {
                str(index) for index in range(window_length)
            }:
                raise ValueError(
                    f"{path}: stored timestep columns disagree with preprocessing widths {allowed}"
                )
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
        for row, seq in enumerate(ids):
            identity = int(seq)
            interval = (int(lows[row]), int(highs[row]))
            if bounds.setdefault(identity, interval) != interval:
                raise ValueError("Prepared windows disagree on split bounds")
            for offset in range(
                int(pads[row]), next(iter(sequences.values())).shape[1]
            ):
                position = int(starts[row]) + offset
                key = (identity, position)
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
                }
                if labels := identities.get(str(identity)):
                    for column, field in (
                        ("sequenceName", "episode_id"),
                        ("groupId", "group_id"),
                    ):
                        if labels.get(field) is not None:
                            record[column] = labels[field]
                if key in rows and rows[key] != record:
                    raise ValueError(
                        "prepared history windows disagree on overlapping positions or split bounds"
                    )
                rows[key] = record
    if not rows:
        raise ValueError("prepared source contains no valid positions")
    frame = pl.DataFrame(list(rows.values())).sort(["sequenceId", "itemPosition"])
    if metadata.source_observations_path:
        from sequifier.io.sequence import DOMAIN_OBSERVATIONS

        coordinates = ["sequenceId", "itemPosition"]
        physical = pl.scan_parquet(
            source.project_root / metadata.source_observations_path
        )
        frame = (
            frame.lazy()
            .join(
                physical,
                on=coordinates,
                how="left",
                validate="1:1",
                maintain_order="left",
            )
            .collect()
        )
        if frame[DOMAIN_OBSERVATIONS].null_count():
            raise ValueError("Source observations do not cover the prepared windows")
    return frame, bounds
