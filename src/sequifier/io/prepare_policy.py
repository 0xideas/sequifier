"""Prepare indexed training views from canonical policy demonstrations."""

import json
from dataclasses import replace
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa

from sequifier.config.depth_layout import (
    DepthLayoutRegistryModel,
    depth_mask_metadata_key,
)
from sequifier.config.metadata import RawInputMetadata, load_dataset_metadata
from sequifier.config.splits import validate_disjoint_roles
from sequifier.io.atomic import atomic_path
from sequifier.io.episode_corpus import EpisodeCorpus
from sequifier.io.feature_fitting import (
    FeatureStatistics,
    FittedColumns,
    FittingRules,
    cast_input_columns,
)
from sequifier.io.indexed_sequences import (
    PREFIX_IMAGE,
    IndexedSequenceWriter,
    IndexedView,
)
from sequifier.io.policy_dataset import EpisodeDecisions, episode_split_assignments
from sequifier.io.preprocessing_workspace import (
    indexed_preparation,
    preprocessing_workspace,
)
from sequifier.io.sample_order import curriculum_columns, curriculum_storage_column
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
    REWARD_COLUMN,
    SUCCESSOR_PREFIX,
    TRANSITION_FLAGS,
    TableEpisodeReference,
    executed_action_columns,
    project_actions,
    scan_tables,
)


def _preparation_inputs(config):
    original = (
        Path(config.project_root) / config.metadata_config_path
        if config.metadata_config_path
        else None
    )
    resolved = config.resolved_source()
    if (
        config.read_format == "sequence"
        and resolved.kind != "trajectory"
        and (resolved.manifest or {}).get("format") != "sequifier.sequence-data"
    ):
        raise ValueError(
            "read_format: sequence requires a sequence manifest or trajectory"
        )
    if config.read_format in {"csv", "parquet"} and (
        resolved.kind != "table"
        or any(
            path.suffix.lower() != f".{config.read_format}" for path in resolved.paths
        )
    ):
        raise ValueError(
            f"read_format: {config.read_format} requires matching table files"
        )
    if original is not None:
        resolved.dependencies.add(original)
    if config.initialize_from:
        resolved.dependencies.add(Path(config.project_root) / config.initialize_from)
    if config.use_precomputed_maps is not None:
        directory = Path(config.project_root) / "configs" / "id_maps"
        for path in directory.glob("*.json"):
            resolved.dependencies.add(path)
    if config.image_sources or config.preconditioning_image:
        from sequifier.io.images import media_files

        locations = [
            source.path for source in config.image_sources.values() if source.path
        ]
        if config.preconditioning_image and config.preconditioning_image.paths:
            locations.extend(config.preconditioning_image.paths.values())
        for location in locations:
            for path in media_files(location, config.project_root):
                resolved.dependencies.add(path)
    return resolved, original


def _validate_unfingerprinted_store(config, destination, provenance):
    hint = "choose a new indexed_output_path"
    if not provenance.get("policy_binding"):
        raise ValueError(f"Published dataset lacks a recorded policy binding; {hint}")
    policy = config.resolved_policy()
    if provenance["policy_binding"] != policy.task_binding.model_dump(mode="json"):
        raise ValueError(f"Published dataset has a different policy binding; {hint}")

    view = IndexedView.load(destination / "0.split.json")
    metadata = view.metadata()
    layout = {**config.split_metadata(), **config.window_metadata()}
    if any(getattr(metadata, key, None) != value for key, value in layout.items()):
        raise ValueError(
            f"Published dataset has a different split or window layout; {hint}"
        )
    if (config.mask_column and config.mask_column != metadata.raw_mask_column) or set(
        config.indexed_passthrough
    ) - set(metadata.indexed_passthrough):
        raise ValueError(
            f"Published dataset has different mask or passthrough settings; {hint}"
        )
    static = [name for name in policy.static_columns if name != PREFIX_IMAGE]
    derived = ["__policy_return"] if policy.return_conditioned else []
    if (
        set(static) != set(metadata.static_columns)
        or (PREFIX_IMAGE in policy.static_columns)
        != bool(metadata.preconditioning_image)
        or derived != metadata.indexed_derived_features
    ):
        raise ValueError(
            f"Published dataset has different static or derived features; {hint}"
        )

    encoder = EpisodeTableEncoder(policy)
    actions = executed_action_columns(policy.actions, encoder.columns)
    view.validate_resources()
    view.validate_item_columns(
        (encoder.columns - set(policy.static_columns))
        | set(actions.values())
        | set(policy.previous_action_inputs)
        | set(config.indexed_passthrough),
        struct_fields={
            name: policy.actions for name in (ACTION_RECORD, PREVIOUS_ACTION_RECORD)
        },
    )


def prepare_policy(config):
    destination = Path(config.project_root) / config.indexed_output_path
    with preprocessing_workspace(
        destination, indexed=True, resume=not config.reprocess
    ) as work:
        if work is None:
            published = json.loads((destination / "dataset.json").read_text())
            provenance = published.get("provenance", {})
            if not provenance.get("policy_binding") or "preparation" not in provenance:
                _validate_unfingerprinted_store(config, destination, provenance)
                return
        resolved, original = _preparation_inputs(config)
        # Omitted layout options must also match datasets published before they
        # were available in policy preparation.
        layout_fields = {
            "read_format",
            "selected_columns",
            "column_data_types",
            "static_data_path",
            "static_categorical_columns",
            "static_real_columns",
            "depth_layouts",
            "image_sources",
            "preconditioning_image",
            "time_alignment_column",
            "curriculum_column",
            "use_precomputed_maps",
        }
        exclusions: dict[str, Any] = dict.fromkeys(
            layout_fields - config.model_fields_set, True
        )
        if "execution" not in config.run.model_fields_set:
            exclusions["run"] = {"execution": True}
        config, identity = indexed_preparation(
            config,
            destination,
            work,
            resolved.dependencies.identity,
            exclude_settings=exclusions,
            action_order=list(config.policy_binding.actions),
            feedback_order=list(config.policy_binding.feedback),
        )
        if work is None:
            return
        with IndexedSequenceWriter(destination, workspace=work) as writer:
            _prepare_policy(config, writer, resolved, original, identity)


def _prepare_policy(config, writer, resolved, original, identity):
    from sequifier.preprocess import (
        SPLIT_VALUE_COLUMN,
        _add_normalized_split_column,
        _normalize_cardinality_config,
        _normalized_split_cutoffs,
        _validate_cardinality_columns,
        _validate_declared_column_roles,
        load_precomputed_id_maps,
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
    images = config.image_sources or (
        frozen.image_sources if frozen is not None else {}
    )
    prefix = config.preconditioning_image or (
        frozen.preconditioning_image if frozen is not None else None
    )
    time_column = config.time_alignment_column or (
        frozen.time_alignment_column if frozen is not None else None
    )
    if (
        config.time_alignment_column
        and frozen is not None
        and config.time_alignment_column != frozen.time_alignment_column
    ):
        raise ValueError(
            "time_alignment_column differs from the original preprocessing metadata"
        )
    policy = config.resolved_policy(resolved)
    media_inputs = {
        key for name in images for key in (f"__image:{name}", f"__image_mask:{name}")
    }
    if config.initialize_from is None and (images or prefix):
        policy = policy.model_copy(
            update={
                "observation_columns": list(
                    dict.fromkeys([*policy.observation_columns, *sorted(media_inputs)])
                ),
                "static_columns": list(
                    dict.fromkeys(
                        [*policy.static_columns, *([PREFIX_IMAGE] if prefix else [])]
                    )
                ),
            }
        )
    layouts = (
        frozen.depth_layouts.relevant_layouts(policy.observation_columns)
        if frozen is not None
        else DepthLayoutRegistryModel()
    )
    if config.depth_layouts:
        if frozen is not None and config.depth_layouts != layouts:
            raise ValueError(
                "depth_layouts differs from the original preprocessing metadata"
            )
        layouts = config.depth_layouts
    if layouts and config.initialize_from is None:
        masks = [depth_mask_metadata_key(name) for name in layouts.root]
        policy = policy.model_copy(
            update={
                "observation_columns": list(
                    dict.fromkeys([*policy.observation_columns, *masks])
                )
            }
        )
    if (
        config.selected_columns is not None
        or config.static_categorical_columns
        or config.static_real_columns
    ):
        declaration = resolved.declaration
        resolved = replace(
            resolved,
            declaration=declaration.model_copy(
                update={
                    "columns": declaration.columns.model_copy(
                        update={
                            "observations": policy.observation_columns,
                            "static_inputs": policy.static_columns,
                        }
                    )
                }
            ),
        )
    if layouts and resolved.kind == "table":
        resolved = _prepare_depth_source(
            resolved, layouts, writer.workspace, policy, config, time_column
        )
    external_media = any(
        source.path for source in config.image_sources.values()
    ) or bool(config.preconditioning_image and config.preconditioning_image.paths)
    if external_media:
        if any(
            name not in config.image_sources or not config.image_sources[name].path
            for name in images
        ) or (
            prefix
            and not (
                config.preconditioning_image and config.preconditioning_image.paths
            )
        ):
            raise ValueError(
                "Use inline or external media consistently in policy preparation"
            )
        if resolved.kind != "table":
            raise ValueError("External policy media requires a table source")
        store = _prepare_media(resolved, images, prefix, time_column, writer.workspace)
        images, prefix = store.sources, store.preconditioning
        resolved = replace(resolved, image_store=store)
    raw_inputs = frozen or policy.raw_inputs or resolved.metadata
    if config.mask_column:
        if raw_inputs and raw_inputs.raw_mask_column not in (None, config.mask_column):
            raise ValueError(
                "mask_column differs from the original preprocessing metadata"
            )
        raw_inputs = (raw_inputs or RawInputMetadata()).model_copy(
            update={"raw_mask_column": config.mask_column}
        )
    policy = policy.model_copy(update={"raw_inputs": raw_inputs})
    if config.initialize_from is None:
        policy = policy.model_copy(
            update={"context_length": max(config.window_lengths)}
        )
    source_episodes = EpisodeCorpus([resolved], policy).view()
    decisions = EpisodeDecisions(source_episodes, policy)
    plan = SplitWindowPlan.from_settings(config)
    fractions = plan.ratios
    split_count = len(config.splits)
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
    if missing := media_inputs - set(observations):
        raise ValueError(f"Policy preparation lacks image inputs: {sorted(missing)}")
    if (prefix is not None) != (PREFIX_IMAGE in policy.static_columns):
        raise ValueError(
            "Policy preparation requires matching preconditioning image metadata and input"
        )
    depth_masks = {depth_mask_metadata_key(name) for name in layouts.root}
    if missing := depth_masks - set(observations):
        raise ValueError(f"Policy preparation lacks depth masks: {sorted(missing)}")
    features = [name for name in observations if name not in media_inputs | depth_masks]
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
    action_columns = executed_action_columns(policy.actions, encoder.columns)
    actions = list(action_columns.values())
    # A shared self-feedback/target column stores the executed action, as in
    # supervised next-step prediction. Policy windows obtain its lag from the
    # separate previous-action record, including at a fragment boundary.
    feedback = [
        name for name, action in policy.previous_action_inputs.items() if name != action
    ]
    temporal = features + actions + feedback
    columns = temporal + static
    if config.selected_columns is not None:
        missing = (
            set(config.selected_columns)
            - set(temporal)
            - media_inputs
            - depth_masks
            - {config.split_column}
        )
        if missing:
            raise ValueError(f"Unknown policy selected_columns: {sorted(missing)}")
    types = {
        name: dtype
        for name, dtype in (config.column_data_types or {}).items()
        if name not in {config.split_column, time_column}
        or not dtype.startswith(("Date", "Datetime"))
    }
    if unknown := set(types) - set(columns):
        raise ValueError(
            f"column_data_types names unknown policy columns: {sorted(unknown)}"
        )
    if frozen is not None and types:
        for name, dtype in types.items():
            stored_dtype = frozen.column_data_types.get(name)
            if stored_dtype is not None and stored_dtype != dtype:
                raise ValueError(
                    f"column_data_types.{name}: {dtype} differs from frozen dtype {stored_dtype}"
                )
    curriculum = curriculum_columns(config.curriculum_column)
    stored_curriculum = [curriculum_storage_column(name) for name in curriculum]
    derived = ["__policy_return"] if policy.return_conditioned else []
    # Transition scalars already have domain values in the canonical table.
    auxiliary = [
        name
        for name in config.indexed_passthrough
        if name not in (REWARD_COLUMN, *TRANSITION_FLAGS)
    ]
    reserved = {
        *columns,
        *derived,
        *schema.names,
        policy.raw_mask_column,
        "__decision",
        DOMAIN_CONTEXT,
        DOMAIN_STATIC,
        DOMAIN_SUCCESSORS,
        DOMAIN_OBSERVATIONS,
    }
    if collisions := reserved.intersection(auxiliary):
        raise ValueError(
            f"Indexed passthrough overlaps policy fields: {sorted(collisions)}"
        )
    if len(columns) != len(set(columns)):
        raise ValueError(
            "Policy inputs, derived histories, and targets must have distinct names"
        )
    categorical: list[str] = [
        *(config.categorical_columns or ()),
        *config.static_categorical_columns,
    ]
    real: list[str] = [*(config.real_columns or ()), *config.static_real_columns]
    if types:
        categorical.extend(name for name, dtype in types.items() if dtype != "Float32")
        real.extend(name for name, dtype in types.items() if dtype == "Float32")
    for name, source_action in (
        *((name, action) for action, name in action_columns.items()),
        *policy.previous_action_inputs.items(),
    ):
        action = policy.actions[source_action]
        (categorical if action.type == "categorical" else real).append(name)
    categorical, real = list(dict.fromkeys(categorical)), list(dict.fromkeys(real))
    validate_disjoint_roles(categorical, real)
    _validate_declared_column_roles(columns, categorical, real)

    precomputed = (
        load_precomputed_id_maps(
            config.project_root, features + static, config.use_precomputed_maps
        )
        if config.use_precomputed_maps is not None
        else {}
    )
    _validate_cardinality_columns(authored_cardinality, features + static, precomputed)
    if frozen is not None and any(
        frozen.id_maps.get(name) != mapping for name, mapping in precomputed.items()
    ):
        raise ValueError(
            "use_precomputed_maps differs from the original preprocessing metadata"
        )

    if collisions := set(feedback) & set(schema.names):
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
        elif column in episode.static_inputs:
            frame = pl.DataFrame(
                {column: [episode.static_inputs[column]] * len(episode.item_positions)}
            )
        else:
            reference = episodes.source(index)
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
            pl.from_arrow(cast_episode_table(table, schema)), policy, action_columns
        )

    raw_schema = _action_features(
        pl.from_arrow(schema.empty_table()), policy, action_columns
    ).schema
    fit_path = writer.workspace / "feature-fit.json"
    # A supervised feature contract may predate the policy targets and feedback.
    fit_columns = (
        columns
        if frozen is None
        else [
            name for name in actions + feedback if name not in frozen.column_data_types
        ]
    )
    metadata = dict(frozen) if frozen is not None else {}
    if frozen is None and fit_path.exists():
        metadata = json.loads(fit_path.read_text())
        metadata["id_maps"] = {
            name: dict(pairs) for name, pairs in metadata["id_maps"].items()
        }
    elif fit_columns:
        declared_maps = policy.categorical_id_maps()
        mappings = {
            name: declared_maps[action]
            for action, name in action_columns.items()
            if action in declared_maps
        }
        mappings.update(
            {name: declared_maps[name] for name in feedback if name in declared_maps}
        )
        mappings.update(precomputed)
        rules = FittingRules(
            precomputed={
                name: mappings[name] for name in fit_columns if name in mappings
            },
            cardinality=cardinality if frozen is None else {},
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
                selected = [name for name in selected if name in fit_columns]
                for name in selected:
                    if layouts.is_deep_column(name):
                        mask_key = depth_mask_metadata_key(
                            layouts.column_to_layout[name]
                        )
                        column = frame[name]
                        layout = layouts.layout_for_column(name)
                        values = _valid_depth_values(
                            column, frame[mask_key].to_list(), layout
                        )
                        feature = pl.Series(
                            name, values, dtype=column.dtype.inner
                        ).to_frame()
                    elif frame[name].dtype.is_nested():
                        raise ValueError(
                            f"{name}: structured features require depth_layouts or metadata_config_path"
                        )
                    else:
                        feature = frame.select(name)
                    features_fit.observe(
                        cast_input_columns(
                            feature,
                            [name],
                            {name: types[name]} if name in types else None,
                        ),
                        [name],
                        rules,
                    )
        maps, stats = features_fit.finish(rules)
        newly_fitted = FittedColumns.resolve(
            fit_columns,
            maps,
            stats,
            raw_schema,
            types={
                name: types.get(name, "Int64" if name in maps else "Float32")
                for name in fit_columns
            },
        )
        for field, values in newly_fitted.metadata(for_resume=True).items():
            metadata[field] = {**metadata.get(field, {}), **values}
        if frozen is None:
            # Preserve numeric categorical keys across the JSON round trip.
            checkpoint = {
                **metadata,
                "id_maps": {
                    name: list(mapping.items()) for name, mapping in maps.items()
                },
            }
            with atomic_path(fit_path) as temporary:
                temporary.write_text(json.dumps(checkpoint))
    stored = columns + [
        name
        for name in derived
        if name in metadata["column_data_types"] and name not in columns
    ]
    fitted = FittedColumns.from_metadata(
        stored, metadata, categorical=categorical, real=real
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
    mask_column = policy.raw_mask_column
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
            rows = pl.read_parquet(
                checkpoint, columns=["sequenceId", "itemPosition", OBSERVATION_MASK]
            )
        else:
            rows = episode_rows(index)
            extra_columns = list(dict.fromkeys([*auxiliary, *curriculum]))
            if extra_columns:
                reference = episodes.source(index)
                if not isinstance(reference, TableEpisodeReference):
                    raise ValueError(
                        "Auxiliary indexed_passthrough and curriculum columns require a table source"
                    )
                extra = reference.decision_columns(
                    episodes[index].item_positions, extra_columns
                )
                rows = rows.hstack(extra.select(auxiliary)) if auxiliary else rows
                if curriculum:
                    for name in curriculum:
                        if (
                            not extra[name].dtype.is_integer()
                            or extra[name].null_count()
                        ):
                            raise ValueError(
                                f"curriculum_column {name!r} requires non-null integers"
                            )
                    rows = rows.hstack(
                        extra.select(
                            pl.col(name).cast(pl.Int64).alias(stored_name)
                            for name, stored_name in zip(
                                curriculum, stored_curriculum, strict=True
                            )
                        )
                    )
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
        if mask_column is None and frozen is None and rows[OBSERVATION_MASK].any():
            mask_column = OBSERVATION_MASK
        first = rows["itemPosition"][0]
        placements = plan.placements(
            first, len(rows), assigned=assigned.get(index), bounds=episode_bounds(index)
        )
        if curriculum and checkpoint.is_file():
            rows = pl.read_parquet(
                checkpoint, columns=["sequenceId", "itemPosition", *stored_curriculum]
            )
        values = {name: rows[name].to_numpy() for name in stored_curriculum}
        writer.add(
            rows,
            {
                split: (placement, placement.curriculum(values, first, sequence_id))
                for split, placement in placements
            },
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
        **config.split_metadata(),
        fit_split_indices=(
            list(range(split_count)) if config.normalize_on_all_data else [0]
        )
        if fit_columns
        else [],
        depth_layouts=layouts,
        **config.window_metadata(),
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
        time_alignment_column=time_column,
        **fitted.metadata(),
        normalize_real_columns=normalize,
        normalize_on_all_data=frozen.normalize_on_all_data
        if frozen
        else config.normalize_on_all_data,
        cardinality_config=cardinality,
        raw_mask_column=mask_column,
        static_columns=static,
        static_features_path="static.parquet" if static_rows else None,
        indexed_derived_features=derived,
        indexed_passthrough=config.indexed_passthrough,
        indexed_curriculum_columns=stored_curriculum,
    )
    writer.publish(
        metadata,
        feature_columns=columns + derived,
        original_metadata=original,
        provenance={
            **resolved.provenance(),
            "preparation": identity,
            "policy_binding": policy.task_binding.model_dump(mode="json"),
        },
    )


def _prepare_depth_source(resolved, layouts, workspace, policy, config, time_column):
    from sequifier.io.depth_preprocess import collapse_depth_table
    from sequifier.trajectory.tables import GROUP_COLUMN, SEQUENCE_NAME_COLUMN

    data = scan_tables(resolved.paths)
    schema = data.collect_schema()
    raw = [
        (name, layout)
        for name, layout in layouts.items()
        if layout.position_column in schema
    ]
    if not raw:
        return resolved
    if len(raw) != 1 or len(layouts.root) != 1:
        raise ValueError("Raw policy preparation accepts one repeated-row depth layout")
    name, layout = raw[0]
    coordinates = ["sequenceId", "itemPosition", layout.position_column]
    required = {
        *policy.observation_columns,
        *policy.static_columns,
        *policy.actions,
        *config.indexed_passthrough,
        *curriculum_columns(config.curriculum_column),
        config.split_column,
        config.mask_column,
        time_column,
        ACTION_RECORD,
        PREVIOUS_ACTION_RECORD,
        CONTEXT_COLUMN,
        OBSERVATION_MASK,
        GROUP_COLUMN,
        SEQUENCE_NAME_COLUMN,
        REWARD_COLUMN,
        *TRANSITION_FLAGS,
    }
    columns = coordinates + [
        field for field in schema if field in required and field not in coordinates
    ]
    ordered = workspace / "raw-depth.parquet"
    destination = workspace / "decision-depth.parquet"
    with atomic_path(ordered) as temporary:
        data.select(columns).sort(coordinates).sink_parquet(temporary)
    with atomic_path(destination) as temporary:
        collapse_depth_table(ordered, temporary, name, layout)
    return replace(resolved, paths=(destination,))


def _prepare_media(resolved, images, prefix, time_column, workspace):
    from sequifier.config.image_config import ImageSourceConfig
    from sequifier.io.images import (
        ImageStore,
        _time_ms,
        item_time_mode,
        validate_sources,
    )

    fields = ["sequenceId", "itemPosition"]
    if time_column:
        fields.append(time_column)
    rows = scan_tables(resolved.paths).select(fields).sort(fields[:2]).collect()
    lengths, item_times, time_index = {}, {}, []
    mode = None
    for sequence_rows in rows.partition_by("sequenceId"):
        sequence = sequence_rows["sequenceId"][0]
        length = len(sequence_rows)
        if sequence_rows["itemPosition"].to_list() != list(range(length)):
            raise ValueError(
                "Image sequences require contiguous itemPosition starting at zero"
            )
        lengths[sequence] = length
        if not time_column:
            continue
        times = []
        for value in sequence_rows[time_column]:
            current = item_time_mode(value)
            if mode not in (None, current):
                raise ValueError(
                    "time_alignment_column must use one time representation"
                )
            mode = current
            times.append(_time_ms(value, absolute=mode == "absolute"))
        if (mode == "relative" and times[0] != 0) or any(
            b <= a for a, b in pairwise(times)
        ):
            raise ValueError(
                "Item times must increase and relative item time must start at zero"
            )
        item_times[sequence] = times
        time_index.extend((sequence, index, stamp) for index, stamp in enumerate(times))
    time_path = str(workspace / "item-times.parquet") if time_column else None
    manifest = validate_sources(
        images,
        prefix,
        lengths,
        str(resolved.project_root),
        item_times=item_times if time_column else None,
        time_mode=mode,
        time_index_path=time_path,
        video_index_dir=str(workspace / "video-indexes"),
    )
    if time_path:
        pl.DataFrame(
            time_index, schema=["sequenceId", "itemPosition", "timeMs"], orient="row"
        ).write_parquet(time_path)
    return ImageStore(
        {
            name: ImageSourceConfig.model_validate(
                {**values, "path": images[name].path}
            )
            for name, values in manifest["image_sources"].items()
        },
        prefix.model_copy(
            update={"signature": manifest["preconditioning_image"]["signature"]}
        )
        if prefix
        else None,
        str(resolved.project_root),
    )


def _action_features(rows, policy, action_columns):
    rows = project_actions(rows, action_columns)
    previous = [
        pl.col(PREVIOUS_ACTION_RECORD)
        .struct.field(action)
        .fill_null(pl.lit(policy.initial_inputs.get(name)))
        .alias(name)
        for name, action in policy.previous_action_inputs.items()
        if name != action
    ]
    result = rows.with_columns(previous)
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
    values = column.to_list()
    flattened = _valid_depth_values(column, masks, layout)
    frame = pl.Series(column.name, flattened, dtype=column.dtype.inner).to_frame()
    encoded = iter(
        frame.select(transform.stored_expression(frame[column.name])).to_series()
    )
    return pl.Series(
        column.name,
        [
            None if value is None else [next(encoded) if valid else 0 for valid in mask]
            for value, mask in zip(values, masks, strict=True)
        ],
        dtype=pl.List(transform.output_dtype),
    )


def _valid_depth_values(column, masks, layout):
    if not isinstance(column.dtype, (pl.List, pl.Array)):
        raise ValueError(f"{column.name}: depth values require a list or array")
    flattened = []
    for value, mask in zip(column.to_list(), masks, strict=True):
        if value is None:
            continue
        if (
            mask is None
            or len(value) != layout.max_depth_length
            or len(mask) != len(value)
        ):
            raise ValueError(
                f"{column.name}: depth values and masks must match depth_layouts"
            )
        flattened.extend(item for item, valid in zip(value, mask, strict=True) if valid)
    return flattened
