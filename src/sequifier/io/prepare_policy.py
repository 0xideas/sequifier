"""Prepare indexed training views from canonical policy demonstrations."""

from pathlib import Path

import polars as pl
import pyarrow as pa

from sequifier.config.depth_layout import (
    DepthLayoutRegistryModel,
    depth_mask_metadata_key,
)
from sequifier.config.metadata import load_dataset_metadata
from sequifier.io.episode_corpus import EpisodeCorpus
from sequifier.io.feature_fitting import FeatureStatistics, FittingRules
from sequifier.io.indexed_sequences import PREFIX_IMAGE, IndexedSequenceWriter
from sequifier.io.policy_dataset import EpisodeDecisions, episode_partition_indices
from sequifier.io.preprocessing import FittedFeatureTransform
from sequifier.io.window_placement import WindowPlacement
from sequifier.special_tokens import SPECIAL_TOKEN_IDS
from sequifier.trajectory.export import EpisodeTableEncoder, cast_episode_table
from sequifier.trajectory.sources import resolve_source
from sequifier.trajectory.tables import (
    ACTION_RECORD,
    PREVIOUS_ACTION_RECORD,
    SUCCESSOR_PREFIX,
)


def prepare_policy(config):
    from sequifier.preprocess import _validate_declared_column_roles

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
    resolved = resolve_source(source)
    source_episodes = EpisodeCorpus([resolved], policy).view()
    decisions = EpisodeDecisions(source_episodes, policy)
    train, validation = episode_partition_indices(
        decisions,
        config.validation_fraction,
        config.run.reproducibility.seed,
        group_identical=True,
    )
    split_by_index = {
        i: split for split, indices in enumerate((train, validation)) for i in indices
    }
    sequence_ids = {name: index for index, name in enumerate(decisions.all_episode_ids)}
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
    encoder = EpisodeTableEncoder(policy)
    episodes = decisions.episodes
    schema = None
    for index, episode in enumerate(source_episodes):
        current = encoder.table(index, episode).schema
        schema = (
            current
            if schema is None
            else pa.unify_schemas([schema, current], promote_options="permissive")
        )
    if schema is None or not len(episodes):
        raise ValueError("Policy preparation requires eligible episodes")
    if collisions := set(actions + list(policy.previous_action_inputs)) & set(
        schema.names
    ):
        raise ValueError(
            f"Action columns collide with recorded fields: {sorted(collisions)}"
        )

    def episode_rows(index):
        episode = episodes[index]
        table = encoder.table(sequence_ids[episode.episode_id], episode)
        return _action_features(
            pl.from_arrow(cast_episode_table(table, schema)), policy
        )

    with IndexedSequenceWriter(destination) as writer:
        if frozen is None:
            rules = FittingRules(
                precomputed=policy.categorical_id_maps(),
                categorical=categorical,
                real=real,
            )
            features_fit = FeatureStatistics()
            types = None
            for index in sorted(train):
                rows = episode_rows(index).head(decisions.lengths[index])
                for frame, selected in ((rows, temporal), (rows.head(1), static)):
                    if nested := [
                        name for name in selected if frame[name].dtype.is_nested()
                    ]:
                        raise ValueError(
                            f"Structured policy inputs require metadata_config_path: {nested}"
                        )
                    if selected:
                        features_fit.observe(frame, selected, rules)
            maps, stats = features_fit.finish(rules)
            classes = {
                name: max(SPECIAL_TOKEN_IDS.user_start, max(mapping.values()) + 1)
                for name, mapping in maps.items()
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

        transforms = {}
        static_rows = []
        for index in range(len(episodes)):
            rows = episode_rows(index)
            record = episodes.metadata(index)
            episode_id = record.episode_id
            split = split_by_index[index]
            if not transforms:
                raw_schema = rows.schema
                for name in columns:
                    dtype = raw_schema[name]
                    if layouts.is_deep_column(name):
                        dtype = dtype.inner
                    transforms[name] = FittedFeatureTransform.for_column(
                        name, dtype, types, maps, stats, normalize, cardinality
                    )
                if types is None:
                    types = {
                        name: transform.dtype for name, transform in transforms.items()
                    }
            rows = rows.with_columns(
                _encoded_observation(rows, name, source_prefix, layouts, transform)
                for name, transform in transforms.items()
                for source_prefix in (
                    ("", SUCCESSOR_PREFIX) if name in features else ("",)
                )
            )
            static_payload = [*static, *([PREFIX_IMAGE] if prefix else [])]
            if static_payload:
                static_rows.append(rows.select("sequenceId", *static_payload).head(1))
                rows = rows.drop(static_payload)
            first = rows["itemPosition"][0]
            placement = WindowPlacement.for_split(
                first,
                first,
                first + len(rows),
                config.window_lengths[split],
                config.window_stride,
            )
            writer.add(
                rows,
                {split: (placement, None)},
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
            raw_mask_column=frozen.raw_mask_column if frozen else None,
            static_columns=static,
            static_features_path="static.parquet" if static_rows else None,
            indexed_derived_features=derived,
        )
        writer.publish(
            metadata,
            feature_columns=columns + derived,
            original_metadata=original,
            provenance=resolved.provenance(source),
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


def _encoded_observation(rows, name, prefix, layouts, transform):
    column = rows[prefix + name]
    layout = layouts.layout_for_column(name)
    if prefix and column.null_count() == len(rows):
        dtype = pl.List(transform.output_dtype) if layout else transform.output_dtype
        return pl.lit(None, dtype=dtype).alias(column.name)
    if layout is None:
        return transform.stored_expression(column, preserve_nulls=bool(prefix))

    mask_key = depth_mask_metadata_key(layouts.column_to_layout[name])
    masks = rows[prefix + mask_key].to_list()
    flattened = []
    values = column.to_list()
    for value, mask in zip(values, masks):
        if value is None:
            continue
        if (
            mask is None
            or len(value) != layout.max_depth_length
            or len(mask) != len(value)
        ):
            raise ValueError(
                f"{column.name}: depth values and masks must match the frozen layout"
            )
        flattened.extend(item for item, valid in zip(value, mask) if valid)
    frame = pl.Series(column.name, flattened, dtype=column.dtype.inner).to_frame()
    encoded = iter(
        frame.select(transform.stored_expression(frame[column.name])).to_series()
    )
    return pl.Series(
        column.name,
        [
            None if value is None else [next(encoded) if valid else 0 for valid in mask]
            for value, mask in zip(values, masks)
        ],
        dtype=pl.List(transform.output_dtype),
    )
