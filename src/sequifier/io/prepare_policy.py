"""Prepare indexed training views from canonical policy demonstrations."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from sequifier.config.depth_layout import (
    DepthLayoutRegistryModel,
    depth_mask_metadata_key,
)
from sequifier.config.metadata import load_dataset_metadata
from sequifier.contract_versions import ContractVersions
from sequifier.helpers import StoredWindowLayout, polars_dtype_from_name
from sequifier.io.indexed_sequences import PREFIX_IMAGE, IndexedSequenceWriter
from sequifier.io.policy_dataset import EpisodeDecisions, episode_partition_indices
from sequifier.special_tokens import SPECIAL_TOKEN_IDS
from sequifier.trajectory.export import import_sequences
from sequifier.trajectory.importers import source_episodes
from sequifier.trajectory.tables import (
    ACTION_RECORD,
    EPISODE_ID_COLUMN,
    PREVIOUS_ACTION_RECORD,
    SUCCESSOR_PREFIX,
    IndexedTable,
    scan_tables,
)


def prepare_policy(config):
    from sequifier.preprocess import (
        _apply_column_statistics,
        _apply_output_type_casting,
        _batch_to_arrays,
        _extract_sequence_windows_for_splits,
        _get_column_statistics,
        _validate_declared_column_roles,
    )

    original = (
        Path(config.project_root) / config.metadata_config_path
        if config.metadata_config_path
        else None
    )
    frozen = load_dataset_metadata(str(original)) if original is not None else None
    cardinality = frozen.cardinality_config if frozen is not None else {}
    normalize = (
        frozen.normalize_real_columns
        if frozen is not None
        else config.normalize_real_columns
    )
    policy = config.resolved_policy()
    source = config.resolved_source()
    decisions = EpisodeDecisions(source_episodes(source, policy), policy)
    train, validation = episode_partition_indices(
        decisions,
        config.validation_fraction,
        config.run.reproducibility.seed,
        group_identical=True,
    )
    split_by_episode = {
        decisions.episode_ids[index]: split
        for split, indices in enumerate((train, validation))
        for index in indices
    }
    episode_records = {
        decisions.episode_ids[index]: decisions.episodes.metadata(index)
        for index in range(len(decisions.episodes))
    }
    fit_lengths = {
        decisions.episode_ids[index]: decisions.lengths[index] for index in train
    }
    widths = [
        StoredWindowLayout(
            width, config.max_target_offset, ContractVersions.STORED_WINDOW_LAYOUT
        )
        for width in config.window_lengths
    ]
    observations, static = policy.observation_columns, policy.static_columns
    static = [name for name in static if name != PREFIX_IMAGE]
    images = frozen.image_sources if frozen is not None else {}
    prefix = frozen.preconditioning_image if frozen is not None else None
    media_inputs = {
        key for name in images for key in (f"__image:{name}", f"__image_mask:{name}")
    }
    if missing := media_inputs - set(observations):
        raise ValueError(f"Policy preparation lacks image inputs: {sorted(missing)}")
    if (prefix is not None) != (PREFIX_IMAGE in policy.static_columns):
        raise ValueError(
            "Policy preparation requires matching preconditioning image metadata and input"
        )
    layouts = (
        frozen.depth_layouts.relevant_layouts(observations)
        if frozen is not None
        else DepthLayoutRegistryModel()
    )
    depth_masks = {depth_mask_metadata_key(name) for name in layouts.root}
    if missing := depth_masks - set(observations):
        raise ValueError(f"Policy preparation lacks depth masks: {sorted(missing)}")
    features = [name for name in observations if name not in media_inputs | depth_masks]
    actions = list(policy.actions)
    temporal = features + actions + list(policy.previous_action_inputs)
    columns = temporal + static
    if len(columns) != len(set(columns)):
        raise ValueError(
            "Policy inputs, derived histories, and targets must have distinct names"
        )
    categorical: list[str] = list(config.categorical_columns or ())
    real: list[str] = list(config.real_columns or ())
    for name, source_action in (
        *((name, name) for name in actions),
        *policy.previous_action_inputs.items(),
    ):
        action = policy.actions[source_action]
        (categorical if action.type == "categorical" else real).append(name)
    categorical, real = list(dict.fromkeys(categorical)), list(dict.fromkeys(real))
    _validate_declared_column_roles(columns, categorical, real)

    destination = Path(config.project_root) / config.output_path
    with (
        TemporaryDirectory(prefix="sequifier-policy-preparation-") as scratch,
        IndexedSequenceWriter(destination) as writer,
    ):
        imported = import_sequences(
            config.model_copy(update={"output_path": str(Path(scratch) / "canonical")})
        )
        rows = scan_tables(tuple(sorted(imported.parent.glob("*.parquet"))))
        if collisions := set(actions + list(policy.previous_action_inputs)) & set(
            rows.collect_schema().names()
        ):
            raise ValueError(
                f"Action columns collide with recorded fields: {sorted(collisions)}"
            )
        table = IndexedTable(rows, "sequenceId")
        if frozen is None:
            maps, stats, classes, types = {}, {}, None, None
            for sequence in table.ranges:
                rows = _action_features(table.load(sequence), policy)
                episode_id = rows[EPISODE_ID_COLUMN][0]
                if episode_id not in fit_lengths:
                    continue
                rows = rows.head(fit_lengths[episode_id])
                for frame, selected in ((rows, temporal), (rows.head(1), static)):
                    if nested := [
                        name for name in selected if frame[name].dtype.is_nested()
                    ]:
                        raise ValueError(
                            f"Structured policy inputs require metadata_config_path: {nested}"
                        )
                    if selected:
                        maps, stats = _get_column_statistics(
                            frame,
                            selected,
                            maps,
                            stats,
                            0,
                            {},
                            categorical_columns=categorical,
                            real_columns=real,
                        )
            for name in actions:
                action = policy.actions[name]
                if action.type == "categorical" and action.values:
                    maps[name] = {
                        value: SPECIAL_TOKEN_IDS.user_start + index
                        for index, value in enumerate(action.values)
                    }
        else:
            maps = frozen.id_maps
            stats = dict(frozen.selected_columns_statistics)
            classes = frozen.n_classes
            types = dict(frozen.column_data_types)
            missing = set(columns) - types.keys()
            if missing:
                raise ValueError(
                    f"Frozen policy metadata lacks columns: {sorted(missing)}"
                )

        static_rows = []
        for sequence in table.ranges:
            rows = _action_features(table.load(sequence), policy)
            episode_id = rows[EPISODE_ID_COLUMN][0]
            if episode_id not in split_by_episode:
                continue
            split = split_by_episode[episode_id]
            for name in features:
                successor = SUCCESSOR_PREFIX + name
                if rows[successor].null_count() == len(rows):
                    continue
                rows = rows.with_columns(
                    _transform_observation(
                        rows,
                        successor,
                        name,
                        layouts,
                        maps,
                        stats,
                        normalize,
                        cardinality,
                    )
                )
            scalar = [
                column for column in columns if not layouts.is_deep_column(column)
            ]
            for name in set(columns) - set(scalar):
                if types is None:
                    raise ValueError("Structured policy inputs require metadata types")
                rows = rows.with_columns(
                    _transform_observation(
                        rows, name, name, layouts, maps, stats, normalize, cardinality
                    ).cast(pl.List(polars_dtype_from_name(types[name])))
                )
            rows, classes, types = _apply_column_statistics(
                rows, scalar, maps, stats, normalize, classes, types, cardinality
            )
            rows = _apply_output_type_casting(rows, scalar, types)
            # Successors must round the same way as the next observed row.
            schema = rows.schema
            rows = rows.with_columns(
                pl.col(SUCCESSOR_PREFIX + name).cast(schema[name]) for name in features
            )
            static_payload = [*static, *([PREFIX_IMAGE] if prefix else [])]
            if static_payload:
                static_rows.append(rows.select("sequenceId", *static_payload).head(1))
                rows = rows.drop(static_payload)
            arrays = _batch_to_arrays(rows, [])
            views = _extract_sequence_windows_for_splits(
                arrays,
                0,
                len(rows),
                sequence,
                widths,
                config.window_stride,
                [],
                [1 - config.validation_fraction, config.validation_fraction],
                "between_sequence",
                config.run.reproducibility.seed,
                {sequence: split},
            )
            record = episode_records[episode_id]
            writer.add(
                rows,
                views,
                episode_id=episode_id,
                group_id=record.group_id,
                content_key=record.content_key,
            )
        if types is None:
            raise ValueError("Policy preparation requires eligible training episodes")
        if static_rows:
            writer.write_static(pl.concat(static_rows, how="vertical_relaxed"))
        derived = []
        if policy.return_conditioned:
            types.setdefault("__policy_return", "Float32")
            stats.setdefault("__policy_return", {"mean": 0.0, "std": 1.0})
            derived.append("__policy_return")
        metadata = dict(
            split_paths=["0.split.json", "1.split.json"],
            indexed_views=["0", "1"],
            fit_split_indices=[0] if frozen is None else [],
            depth_layouts=layouts,
            window_length=config.window_lengths[0],
            window_lengths=list(config.window_lengths),
            max_target_offset=config.max_target_offset,
            window_stride=config.window_stride,
            image_sources={
                name: image.compatibility_signature() for name, image in images.items()
            },
            preconditioning_image=(
                prefix.model_copy(
                    update={
                        "view_names": sorted(prefix.views),
                        "paths": None,
                        "signature": None,
                    }
                )
                if prefix
                else None
            ),
            indexed_inline_media=bool(images or prefix),
            time_alignment_column=frozen.time_alignment_column if frozen else None,
            column_data_types=types,
            id_maps=maps,
            n_classes=classes,
            selected_columns_statistics={
                column: {key: values[key] for key in ("mean", "std")}
                for column, values in stats.items()
            },
            normalize_real_columns=normalize,
            normalize_on_all_data=frozen.normalize_on_all_data if frozen else False,
            cardinality_config=cardinality,
            static_columns=static,
            static_features_path="static.parquet" if static_rows else None,
            indexed_derived_features=derived,
        )
        writer.publish(
            metadata,
            feature_columns=columns + derived,
            original_metadata=original,
            provenance=json.loads(imported.read_text())["provenance"],
        )


def _action_features(rows, policy):
    targets = [
        pl.col(ACTION_RECORD).struct.field(name).alias(name) for name in policy.actions
    ]
    previous = [
        pl.col(PREVIOUS_ACTION_RECORD)
        .struct.field(action)
        .fill_null(pl.lit(policy.initial_inputs.get(name)))
        .alias(name)
        for name, action in policy.previous_action_inputs.items()
    ]
    result = rows.with_columns(*targets, *previous)
    if any(result[name].null_count() for name in policy.previous_action_inputs):
        raise ValueError(
            "Policy preparation requires initial_inputs for missing lagged actions"
        )
    return result


def _transform_observation(
    rows, source, feature, layouts, maps, stats, normalize, cardinality
):
    """Encode an observed or successor value using the declared feature transform."""
    from sequifier.preprocess import _apply_column_statistics

    values = rows[source].to_list()
    layout = layouts.layout_for_column(feature)
    if layout is None:
        masks: list[list[bool] | None] = [None] * len(values)
    else:
        name = layouts.column_to_layout[feature]
        mask_column = depth_mask_metadata_key(name)
        if source != feature:
            mask_column = SUCCESSOR_PREFIX + mask_column
        masks = rows[mask_column].to_list()
    flattened = []
    for value, mask in zip(values, masks):
        if value is None:
            continue
        if layout is None:
            flattened.append(value)
        else:
            if (
                mask is None
                or len(value) != layout.max_depth_length
                or len(mask) != len(value)
            ):
                raise ValueError(
                    f"{feature}: depth values and masks must match the frozen layout"
                )
            flattened.extend(item for item, valid in zip(value, mask) if valid)
    transformed = []
    if flattened:
        frame = pl.DataFrame({feature: flattened})
        transformed, _, _ = _apply_column_statistics(
            frame, [feature], maps, stats, normalize, cardinality_config=cardinality
        )
        transformed = transformed[feature].to_list()
    encoded = iter(transformed)
    output = []
    for value, mask in zip(values, masks):
        if value is None:
            output.append(None)
        elif layout is None:
            output.append(next(encoded))
        else:
            assert mask is not None  # Validated in the first pass above.
            output.append([next(encoded) if valid else 0 for valid in mask])
    return pl.Series(source, output)
