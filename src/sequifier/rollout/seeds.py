"""Resolve a physical seed item independently of prepared window ordering."""

from pathlib import Path

import numpy as np
import torch


def seed_origin(config, sequence, position):
    from sequifier.helpers import normalize_path

    view = getattr(config, "_indexed_view", None)
    dataset = (
        view.manifest_path
        if view is not None and config.input_layout != "raw"
        else normalize_path(config.data_path, config.project_root)
    )
    split = getattr(config, "split", None)
    if split is None and view is not None and config.input_layout != "raw":
        identifier = view.view["view"]
        names = view.metadata().split_names
        if identifier == "predict":
            split = identifier
        elif int(identifier) < len(names):
            split = names[int(identifier)]
    return {
        "dataset": str(Path(dataset).resolve()),
        "split": split,
        "sequenceId": sequence,
        "itemPosition": int(position),
    }


def indexed_history(view, seed, width):
    sequence, position = seed.sequenceId, seed.itemPosition
    bounds = view.split_bounds()
    if str(sequence) not in view.dataset["sequences"] or (
        view.view["view"] != "predict"
        and (
            sequence not in bounds
            or not bounds[sequence][0] <= position < bounds[sequence][1]
        )
    ):
        raise ValueError("Rollout seed coordinate is outside the selected split")
    rows = view.read(sequence, position - width + 1, position + 1)
    if rows.is_empty() or rows["itemPosition"][-1] != position:
        raise ValueError("Rollout seed coordinate is outside the sequence")
    return rows


def forecast_coordinate_seed(config, dataset, dtypes, codec):
    import polars as pl

    from sequifier.infer import apply_inference_column_types, subset_to_input_columns
    from sequifier.io.pt_payload import tabular_tensor_batch
    from sequifier.io.static_features import table_for_config

    seed = config.rollout.seed
    width = config.window_view.context_length
    if config._indexed_view is not None:
        rows = indexed_history(config._indexed_view, seed, width)
        history = rows.select(config.input_columns).to_dicts()
    else:
        positions = {}
        owned = False
        for data in dataset:
            if isinstance(data, pl.DataFrame):
                data = apply_inference_column_types(
                    subset_to_input_columns(data, config.input_columns), config
                )
                data = tabular_tensor_batch(
                    data,
                    dtypes,
                    config.input_columns,
                    config.storage_layout.window_length,
                    sort_rows=False,
                )
            for i in (
                (data.sequence_ids == seed.sequenceId).nonzero().flatten().tolist()
            ):
                start = int(data.start_item_positions[i])
                low = start + int(data.left_pad_lengths[i])
                end = start + config.storage_layout.window_length
                owned |= (
                    max(low, int(data.split_start_item_positions[i]))
                    <= seed.itemPosition
                    < min(end, int(data.split_end_item_positions[i]))
                )
                for pos in range(
                    max(low, seed.itemPosition - width + 1),
                    min(end, seed.itemPosition + 1),
                ):
                    record = {
                        c: data.sequences[c][i, pos - start].item()
                        for c in config.input_columns
                    }
                    if pos in positions and positions[pos] != record:
                        raise ValueError("Prepared windows disagree at rollout seed")
                    positions[pos] = record
        if not owned or seed.itemPosition not in positions:
            raise ValueError("Rollout seed coordinate is outside the selected split")
        first = seed.itemPosition
        while first - 1 in positions:
            first -= 1
        history = [positions[p] for p in range(first, seed.itemPosition + 1)]
    padding = width - len(history)
    inputs = {
        c: np.asarray([[0] * padding + [r[c] for r in history]])
        for c in config.input_columns
    }
    if config.static_columns:
        static = table_for_config(config)
        assert static is not None
        inputs.update(
            {
                c: value.numpy()
                for c, value in static.lookup(torch.tensor([seed.sequenceId])).items()
            }
        )
    return codec.seed_from_model(
        inputs,
        np.asarray([[False] * padding + [True] * len(history)]),
        config.input_columns,
        config.static_columns,
    )
