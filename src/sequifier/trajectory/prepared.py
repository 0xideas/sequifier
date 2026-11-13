"""Reconstruct split-owned decisions from overlapping prepared histories."""

from pathlib import Path

import polars as pl

from sequifier.config.metadata import load_dataset_metadata
from sequifier.config.policy_train_config import SequenceSourceConfig


def prepared_table(
    paths: tuple[Path, ...], source: SequenceSourceConfig
) -> pl.DataFrame:
    """Recover unique positions from existing windows, validating overlaps.

    Split target bounds select decisions. A preceding action may be recovered
    from stored context, but observations outside the split are not targets.
    """
    if source.metadata_config_path is None:
        raise ValueError("prepared histories require metadata_config_path")
    from sequifier.helpers import PANDAS_TO_TORCH_TYPES, read_data
    from sequifier.io.pt_payload import load_pt_payload, tabular_tensor_batch

    metadata = load_dataset_metadata(source.metadata_config_path)
    from sequifier.config.depth_layout import depth_mask_metadata_key

    rows = {}
    for path in paths:
        if path.suffix.lower() == ".pt":
            batch = load_pt_payload(
                path, layouts=metadata.depth_layouts, n_classes=metadata.n_classes
            )
        else:
            data = read_data(str(path), path.suffix.lower().lstrip("."))
            batch = tabular_tensor_batch(
                data,
                {
                    column: PANDAS_TO_TORCH_TYPES[dtype]
                    for column, dtype in metadata.column_data_types.items()
                },
                sorted(set(data["inputCol"].to_list())),
                metadata.window_length,
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
                if key in rows and rows[key] != record:
                    raise ValueError(
                        "prepared history windows disagree on overlapping positions or split bounds"
                    )
                rows[key] = record
    if not rows:
        raise ValueError("prepared source contains no valid positions")
    return pl.DataFrame(list(rows.values())).sort(["sequenceId", "itemPosition"])
