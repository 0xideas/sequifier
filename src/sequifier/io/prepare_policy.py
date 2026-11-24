"""Prepare indexed training views from canonical policy demonstrations."""

import json
from functools import cache
from pathlib import Path

import polars as pl
import pyarrow as pa

from sequifier.config.depth_layout import (
    DepthLayoutRegistryModel,
    depth_mask_metadata_key,
)
from sequifier.config.metadata import load_dataset_metadata
from sequifier.config.splits import validate_disjoint_roles
from sequifier.io.atomic import atomic_path
from sequifier.io.episode_corpus import EpisodeCorpus
from sequifier.io.feature_fitting import FeatureStatistics, FittedColumns, FittingRules
from sequifier.io.indexed_sequences import PREFIX_IMAGE, IndexedSequenceWriter
from sequifier.io.policy_dataset import EpisodeDecisions, episode_split_assignments
from sequifier.io.preprocessing_workspace import preprocessing_workspace
from sequifier.io.sequence import (
    CONTEXT_COLUMN,
    CONTEXT_RECORDS,
    DOMAIN_CONTEXT,
    DOMAIN_OBSERVATIONS,
    DOMAIN_STATIC,
    DOMAIN_SUCCESSORS,
    OBSERVATION_MASK,
    with_domain_record,
)
from sequifier.io.window_placement import (
    PredictionAlignment,
    SplitWindowPlan,
    cutoff_intervals,
    split_intervals,
)
from sequifier.special_tokens import SPECIAL_TOKEN_IDS
from sequifier.trajectory.export import EpisodeTableEncoder
from sequifier.trajectory.table_schema import cast_episode_table
from sequifier.trajectory.tables import (
    ACTION_RECORD,
    PREVIOUS_ACTION_RECORD,
    SUCCESSOR_PREFIX,
)


def prepare_policy(config):
    destination = Path(config.project_root) / config.indexed_output_path
    with preprocessing_workspace(
        destination, indexed=True, resume=not config.reprocess
    ) as work:
        if work is not None:
            with IndexedSequenceWriter(destination, workspace=work) as writer:
                _prepare_policy(config, writer)


def _prepare_policy(config, writer):
    from sequifier.preprocess import (
        SPLIT_VALUE_COLUMN,
        _add_normalized_split_column,
        _normalize_cardinality_config,
        _normalized_split_cutoffs,
        _validate_cardinality_columns,
        _validate_declared_column_roles,
    )

    original = (
        Path(config.project_root) / config.metadata_config_path
        if config.metadata_config_path
        else None
    )
    frozen = load_dataset_metadata(str(original)) if original is not None else None
    authored_cardinality = _normalize_cardinality_config(config.cardinality_config)
    cardinality = authored_cardinality
    if frozen is not None:
        persisted = _normalize_cardinality_config(frozen.cardinality_config)
        if cardinality and cardinality != persisted:
            raise ValueError("cardinality_config must match metadata_config_path")
        cardinality = persisted
    normalize = (
        frozen.normalize_real_columns
        if frozen is not None
        else config.normalize_real_columns
    )
    resolved = config.resolved_source()
    policy = config.resolved_policy(resolved)
    if frozen is not None:
        policy = policy.model_copy(update={"raw_inputs": frozen})
    if config.initialize_from is None:
        policy = policy.model_copy(
            update={"context_length": max(config.window_lengths)}
        )
    source_episodes = EpisodeCorpus([resolved], policy).view()
    decisions = EpisodeDecisions(source_episodes, policy)
    fractions = tuple(config.split_ratios or ())
    split_count = len(config.splits)
    plan = SplitWindowPlan(
        config.window_lengths,
        config.resolved_window_stride,
        fractions,
        PredictionAlignment.from_settings(config),
    )
    assigned = {}
    if config.split_method == "between_sequence":
        assigned = dict(
            enumerate(
                episode_split_assignments(
                    decisions,
                    list(fractions),
                    config.run.reproducibility.seed,
                    group_identical=True,
                )
            )
        )
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
    # The executed action always has its own record. Expose a scalar training
    # column as well when its name is free in the model feature namespace.
    actions = _scalar_actions(policy)
    # A shared self-feedback/target column stores the executed action, as in
    # supervised next-step prediction. Policy windows obtain its lag from the
    # separate previous-action record, including at a fragment boundary.
    feedback = [
        name for name, action in policy.previous_action_inputs.items() if name != action
    ]
    temporal = features + actions + feedback
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
    validate_disjoint_roles(categorical, real)
    _validate_declared_column_roles(columns, categorical, real)

    _validate_cardinality_columns(authored_cardinality, features + static, {})

    encoder = EpisodeTableEncoder(policy)
    episodes = decisions.episodes
    schema_path = writer.workspace / "episode-schema.arrow"
    if schema_path.exists():
        schema = pa.ipc.read_schema(pa.BufferReader(schema_path.read_bytes()))
    else:
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
        with atomic_path(schema_path) as temporary:
            temporary.write_bytes(schema.serialize().to_pybytes())
    if collisions := set(actions + feedback) & set(schema.names):
        raise ValueError(
            f"Action columns collide with recorded fields: {sorted(collisions)}"
        )

    cutoffs = _normalized_split_cutoffs(config.split_values)

    @cache
    def episode_bounds(index):
        if cutoffs is None:
            return None
        episode = episodes[index]
        column = config.split_column
        if column in episode.observations:
            frame = pl.DataFrame({column: episode.observations[column]})
        else:
            reference = episodes.source(index)
            from sequifier.trajectory.tables import TableEpisodeReference

            if not isinstance(reference, TableEpisodeReference):
                raise ValueError(f"Missing recorded split_column {column!r}")
            frame = reference.decision_columns(episode.item_positions, [column])
        values = _add_normalized_split_column(
            frame, column, config.split_values, episode.episode_id
        )[SPLIT_VALUE_COLUMN]
        if values.null_count():
            raise ValueError("split_column must contain a value for every observation")
        return cutoff_intervals(values.to_numpy(), cutoffs)

    if config.split_method == "within_sequence":
        plan.validate_training_share(fractions)
    else:
        counts = [0] * split_count
        for index in range(len(episodes)):
            if config.split_method == "between_sequence":
                counts[assigned[index]] += episodes.metadata(index).length
            else:
                bounds = episode_bounds(index)
                assert bounds is not None
                for split, (low, high) in enumerate(bounds):
                    counts[split] += high - low
        plan.validate_training_share(counts)

    def episode_rows(index):
        episode = episodes[index]
        table = encoder.table(decisions.sequence_ids[index], episode)
        return _action_features(
            pl.from_arrow(cast_episode_table(table, schema)), policy
        )

    derived = ["__policy_return"] if policy.return_conditioned else []
    raw_schema = _action_features(pl.from_arrow(schema.empty_table()), policy).schema
    fit_path = writer.workspace / "feature-fit.json"
    if frozen is None and fit_path.exists():
        metadata = json.loads(fit_path.read_text())
        metadata["id_maps"] = {
            name: dict(pairs) for name, pairs in metadata["id_maps"].items()
        }
        fitted = FittedColumns.from_metadata(columns, metadata)
    elif frozen is None:
        rules = FittingRules(
            precomputed={
                name: mapping
                for name, mapping in policy.categorical_id_maps().items()
                if name in actions + feedback
            },
            cardinality=cardinality,
            categorical=categorical,
            real=real,
        )
        features_fit = FeatureStatistics()
        for index in range(len(episodes)):
            if not config.normalize_on_all_data and assigned.get(index, 0) != 0:
                continue
            rows = episode_rows(index)
            length = decisions.lengths[index]
            if not config.normalize_on_all_data and index not in assigned:
                bounds = episode_bounds(index) or split_intervals(len(rows), fractions)
                length = min(length, bounds[0][1])
            rows = rows.head(length)
            if rows.is_empty():
                continue
            for frame, selected in (
                (rows.filter(~pl.col(OBSERVATION_MASK)), features),
                (rows, actions + feedback),
                (rows.head(1), static),
            ):
                if nested := [
                    name for name in selected if frame[name].dtype.is_nested()
                ]:
                    raise ValueError(
                        f"Structured policy inputs require metadata_config_path: {nested}"
                    )
                if selected:
                    features_fit.observe(frame, selected, rules)
        maps, stats = features_fit.finish(rules)
        fitted = FittedColumns.resolve(columns, maps, stats, raw_schema)
        metadata = fitted.metadata(for_resume=True)
        # Preserve numeric categorical keys across the JSON round trip.
        metadata["id_maps"] = {
            name: list(mapping.items()) for name, mapping in maps.items()
        }
        with atomic_path(fit_path) as temporary:
            temporary.write_text(json.dumps(metadata))
    else:
        stored = columns + [
            name
            for name in derived
            if name in frozen.column_data_types and name not in columns
        ]
        fitted = FittedColumns.from_metadata(
            stored, dict(frozen), categorical=categorical, real=real
        )

    transforms = {}
    for name in columns:
        dtype = raw_schema[name]
        if layouts.is_deep_column(name):
            if not isinstance(dtype, (pl.List, pl.Array)):
                raise ValueError(f"{name}: depth values require a list or array")
            dtype = dtype.inner
        transforms[name] = fitted.column_transform(name, dtype, normalize, cardinality)
    static_rows = []
    static_payload = [*static, *([PREFIX_IMAGE] if prefix else [])]
    for index in range(len(episodes)):
        record = episodes.metadata(index)
        episode_id = record.episode_id
        sequence_id = decisions.sequence_ids[index]
        checkpoint = writer.item_checkpoint(sequence_id)
        static_path = writer.workspace / f"static-{sequence_id}.parquet"
        if checkpoint.is_file():
            if static_payload:
                static_rows.append(pl.read_parquet(static_path))
            rows = pl.read_parquet(checkpoint, columns=["sequenceId", "itemPosition"])
        else:
            rows = episode_rows(index)
            rows = with_domain_record(rows, features, DOMAIN_OBSERVATIONS)
            rows = with_domain_record(rows, static, DOMAIN_STATIC)
            rows = with_domain_record(
                rows, [SUCCESSOR_PREFIX + name for name in features], DOMAIN_SUCCESSORS
            )
            rows = rows.with_columns(
                _encoded_observation(rows, name, source_prefix, layouts, transform)
                for name, transform in transforms.items()
                for source_prefix in (
                    ("", SUCCESSOR_PREFIX) if name in features else ("",)
                )
            )
            rows = _mask_features(rows, features, fitted, layouts)
            rows = _mask_features(
                rows, features, fitted, layouts, prefix=SUCCESSOR_PREFIX
            )
            rows = rows.with_columns(pl.col(CONTEXT_COLUMN).alias(DOMAIN_CONTEXT))
            if payload := rows[CONTEXT_COLUMN][0]:
                context = CONTEXT_RECORDS.validate_json(payload)
                history = pl.DataFrame([step.observation for step in context])
                history = history.with_columns(
                    _encoded_observation(history, name, "", layouts, transforms[name])
                    for name in features
                )
                history = _mask_features(
                    history.with_columns(
                        pl.Series(OBSERVATION_MASK, [step.masked for step in context])
                    ),
                    features,
                    fitted,
                    layouts,
                ).drop(OBSERVATION_MASK)
                encoded = tuple(
                    step.model_copy(update={"observation": row})
                    for step, row in zip(context, history.to_dicts(), strict=True)
                )
                rows = rows.with_columns(
                    pl.Series(
                        CONTEXT_COLUMN,
                        [
                            CONTEXT_RECORDS.dump_json(encoded).decode(),
                            *([None] * (len(rows) - 1)),
                        ],
                    )
                )
            if static_payload:
                static_row = rows.select(
                    "sequenceId", *static_payload, *([DOMAIN_STATIC] if static else [])
                ).head(1)
                static_rows.append(static_row)
                with atomic_path(static_path) as temporary:
                    static_row.write_parquet(temporary)
                rows = rows.drop(static_payload + ([DOMAIN_STATIC] if static else []))
        first = rows["itemPosition"][0]
        placements = plan.placements(
            first, len(rows), assigned=assigned.get(index), bounds=episode_bounds(index)
        )
        writer.add(
            rows,
            {split: (placement, None) for split, placement in placements},
            episode_id=episode_id,
            group_id=record.group_id,
            content_key=record.content_key,
        )
    if static_rows:
        writer.write_static(pl.concat(static_rows, how="vertical_relaxed"))
    for name in derived:
        fitted.col_types.setdefault(name, "Float32")
        fitted.statistics.setdefault(name, {"mean": 0.0, "std": 1.0})
    metadata = dict(
        split_paths=[f"{i}.split.json" for i in range(split_count)],
        indexed_views=list(map(str, range(split_count))),
        split_names=list(config.splits),
        fit_split_indices=(
            list(range(split_count)) if config.normalize_on_all_data else [0]
        )
        if frozen is None
        else [],
        depth_layouts=layouts,
        window_length=config.window_lengths[0],
        window_lengths=list(config.window_lengths),
        max_target_offset=config.max_target_offset,
        prediction_length=config.prediction_length,
        target_offset=config.target_offset,
        prediction_aligned_splits=config.prediction_aligned_splits,
        prediction_alignment_steps=config.prediction_alignment_steps,
        window_stride=config.resolved_window_stride,
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
        **fitted.metadata(),
        normalize_real_columns=normalize,
        normalize_on_all_data=frozen.normalize_on_all_data
        if frozen
        else config.normalize_on_all_data,
        cardinality_config=cardinality,
        raw_mask_column=policy.raw_mask_column,
        static_columns=static,
        static_features_path="static.parquet" if static_rows else None,
        indexed_derived_features=derived,
    )
    writer.publish(
        metadata,
        feature_columns=columns + derived,
        original_metadata=original,
        provenance={
            **resolved.provenance(),
            "policy_binding": policy.task_binding.model_dump(mode="json"),
        },
    )


def _scalar_actions(policy):
    inputs = set(policy.observation_columns + policy.static_columns)
    return [name for name in policy.actions if name not in inputs]


def _action_features(rows, policy):
    targets = [
        pl.col(ACTION_RECORD).struct.field(name).alias(name)
        for name in _scalar_actions(policy)
    ]
    previous = [
        pl.col(PREVIOUS_ACTION_RECORD)
        .struct.field(action)
        .fill_null(pl.lit(policy.initial_inputs.get(name)))
        .alias(name)
        for name, action in policy.previous_action_inputs.items()
        if name != action
    ]
    result = rows.with_columns(*targets, *previous)
    if any(
        name not in policy.initial_inputs
        and rows[PREVIOUS_ACTION_RECORD].struct.field(action).null_count()
        for name, action in policy.previous_action_inputs.items()
    ):
        raise ValueError(
            "Policy preparation requires feedback initial values for missing lagged actions"
        )
    return result


def _mask_features(rows, features, fitted, layouts, *, prefix=""):
    masked = pl.col(OBSERVATION_MASK)
    if prefix:
        masked = masked.shift(-1, fill_value=False)
    expressions = []
    for name in features:
        column = prefix + name
        fill = SPECIAL_TOKEN_IDS.mask if name in fitted.id_maps else 0
        dtype = rows.schema[column]
        value = (
            pl.col(column).list.eval(pl.element() * 0 + fill)
            if layouts.is_deep_column(name)
            else pl.lit(fill)
        )
        expressions.append(
            pl.when(masked & pl.col(column).is_not_null())
            .then(value.cast(dtype))
            .otherwise(pl.col(column))
            .alias(column)
        )
    return rows.with_columns(expressions)


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
