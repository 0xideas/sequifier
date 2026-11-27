"""Resolve a physical seed item independently of prepared window ordering."""

from pathlib import Path

import polars as pl

from sequifier.io.sequence import DOMAIN_OBSERVATIONS, DOMAIN_STATIC, source_fields
from sequifier.trajectory.schema import SeedHistory


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
    start = position - width + 1
    if view.view["view"] != "predict":
        for first, stop in view.covered_intervals(sequence, *bounds[sequence]):
            if first <= position < stop:
                start = max(start, first)
                break
        else:
            raise ValueError("Rollout seed coordinate is outside the selected split")
    rows = view.read(sequence, start, position + 1)
    if rows.is_empty() or rows["itemPosition"][-1] != position:
        raise ValueError("Rollout seed coordinate is outside the sequence")
    return rows


def forecast_action_columns(config):
    provenance = (
        config._indexed_view.dataset.get("provenance", {})
        if config._indexed_view
        else (getattr(config.dataset_metadata, "provenance", None) or {})
    )
    return provenance.get("action_columns", {})


def forecast_source_seed(config, sequence, rows):
    width = config.window_view.context_length
    if config.static_columns:
        path = (
            config._indexed_view.static_path
            if config._indexed_view
            else Path(config.project_root) / config.static_features_path
        )
        static = (
            pl.scan_parquet(path).filter(pl.col("sequenceId") == sequence).collect()
        )
        if len(static) != 1:
            raise ValueError(
                "Missing or ambiguous source static values for rollout seed"
            )
        values = static.select(
            source_fields(static, DOMAIN_STATIC, config.static_columns)
        ).row(0, named=True)
    else:
        values = {}
    observations = rows.select(
        source_fields(rows, DOMAIN_OBSERVATIONS, config.input_columns)
    ).to_dicts()
    from sequifier.trajectory.tables import observation_mask

    metadata = (
        config._indexed_view.metadata()
        if config._indexed_view
        else config.dataset_metadata
    )
    mask_column = metadata.raw_mask_column
    masks = rows.select(observation_mask(rows, mask_column).alias("mask")).to_series()
    masks = masks.to_list() if len(masks) == len(rows) else [False] * len(rows)
    mapping = forecast_action_columns(config)
    for name in set(config.input_columns).intersection(mapping.values()):
        for row in observations:
            if row[name] is None:
                row[name] = (
                    metadata.selected_columns_statistics[name]["mean"]
                    if metadata.normalize_real_columns
                    else 0.0
                )
    targets = config.target_columns
    mapping = {**{field: field for field in mapping.values()}, **mapping}
    actions = None
    if targets and set(targets) <= mapping.keys():
        domain = rows.schema[DOMAIN_OBSERVATIONS].to_schema()
        fields = [
            pl.col(DOMAIN_OBSERVATIONS).struct.field(mapping[name]).alias(name)
            if mapping[name] in domain
            else pl.col(mapping[name]).alias(name)
            for name in targets
            if mapping[name] in domain or mapping[name] in rows.columns
        ]
        if len(fields) == len(targets):
            actions = [
                row if all(v is not None for v in row.values()) else None
                for row in rows.select(fields).to_dicts()
            ]
    return SeedHistory.from_rows(
        observations, config.input_columns, width, values, masks, actions
    )


def forecast_window_seed(config, sequence, start, stop):
    view = config._indexed_view
    if view is not None:
        rows = view.read(sequence, start, stop)
    else:
        path = config.dataset_metadata.source_observations_path
        if not path:
            raise ValueError("Forecast rollout requires prepared source observations")
        rows = (
            pl.scan_parquet(Path(config.project_root) / path)
            .filter(
                (pl.col("sequenceId") == sequence)
                & pl.col("itemPosition").is_between(start, stop - 1)
            )
            .sort("itemPosition")
            .collect()
        )
    if rows["itemPosition"].to_list() != list(range(start, stop)):
        raise ValueError("Source observations do not cover the rollout window")
    return forecast_source_seed(config, sequence, rows)


def forecast_coordinate_seed(config, dataset):
    seed = config.rollout.seed
    width = config.window_view.context_length
    if config._indexed_view is not None:
        rows = indexed_history(config._indexed_view, seed, width)
        return forecast_source_seed(config, seed.sequenceId, rows)

    # Stored windows own a bounded interval and may include preceding context.
    # Only their coordinates are needed to select physical rows.
    positions = set()
    owned = False
    for data in dataset:
        if isinstance(data, pl.DataFrame):
            bounds = (
                data.filter(pl.col("sequenceId") == seed.sequenceId)
                .select(
                    "startItemPosition",
                    "leftPadLength",
                    "splitStartItemPosition",
                    "splitEndItemPosition",
                )
                .unique()
                .iter_rows()
            )
        else:
            selected = data.sequence_ids == seed.sequenceId
            bounds = zip(
                *(
                    values[selected].tolist()
                    for values in (
                        data.start_item_positions,
                        data.left_pad_lengths,
                        data.split_start_item_positions,
                        data.split_end_item_positions,
                    )
                ),
                strict=True,
            )
        for start, padding, split_start, split_end in bounds:
            low = start + padding
            end = start + config.storage_layout.window_length
            owned |= max(low, split_start) <= seed.itemPosition < min(end, split_end)
            positions.update(
                range(
                    max(low, seed.itemPosition - width + 1),
                    min(end, seed.itemPosition + 1),
                )
            )
    if not owned:
        raise ValueError("Rollout seed coordinate is outside the selected split")
    first = seed.itemPosition
    while first - 1 in positions:
        first -= 1
    return forecast_window_seed(config, seed.sequenceId, first, seed.itemPosition + 1)
