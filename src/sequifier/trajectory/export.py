"""Atomically export canonical episodes as checksummed raw Parquet shards."""

import json
import os
from pathlib import Path

import pyarrow as pa

from sequifier.io.atomic import atomic_path
from sequifier.io.preprocessing_workspace import preprocessing_workspace
from sequifier.io.sequence import (
    TransitionColumns,
    register_sequence_name,
    seed_metadata,
)
from sequifier.trajectory.importers import source_episodes
from sequifier.trajectory.row_layout import ACTION_METADATA, TrajectoryRowLayout
from sequifier.trajectory.sequence_output import arrival_table, publish_sequences
from sequifier.trajectory.table_schema import cast_episode_table
from sequifier.trajectory.tables import SEQUENCE_NAME_COLUMN


def episode_table(sequence, episode, layout, policy, *, first_position=None):
    transitions = episode.transitions or TransitionColumns()
    first_position = (
        episode.first_position if first_position is None else first_position
    )
    rows = []
    for offset in range(len(episode.actions) + 1):
        action = episode.incoming_actions[offset]
        step = transitions.at(offset - 1) if offset else {}
        rows.append(
            layout.row(
                {
                    "sequenceId": sequence,
                    "itemPosition": first_position + offset,
                    "sequenceName": episode.episode_id,
                    "groupId": episode.group_id,
                },
                episode.observation(offset),
                episode.static_inputs,
                action=action,
                reward=step.get("rewards"),
                terminated=step.get("terminated"),
                truncated=step.get("truncated"),
                masked=episode.observation_masks[offset],
            )
        )
    if episode.collection is not None:
        for row, diagnostics in zip(rows, (None, *episode.collection), strict=True):
            row["__collection"] = diagnostics
    return arrival_table(rows, layout, policy)


def import_sequences(config) -> Path:
    """Commit canonical columnar episodes as checksummed raw Parquet shards."""
    destination = (Path(config.project_root) / config.output_path).absolute()
    with preprocessing_workspace(
        destination, indexed=False, resume=not config.reprocess
    ) as work:
        if work is not None:
            _write_import(config, destination, work)
    return destination / "manifest.json"


def _write_import(config, destination, work):
    import pyarrow.parquet as pq

    resolved_source = config.resolved_source()
    policy = config.resolved_policy(resolved_source)
    if (
        config.mask_column
        and not resolved_source.declaration.mask_column
        and not (resolved_source.manifest or {}).get("mask_column")
        and resolved_source.kind == "table"
    ):
        from dataclasses import replace

        resolved_source = replace(
            resolved_source,
            declaration=resolved_source.declaration.model_copy(
                update={"mask_column": config.mask_column}
            ),
        )
    metadata = resolved_source.metadata
    if config.initialize_from is None and metadata is not None:
        policy = policy.model_copy(
            update={
                "context_length": max(
                    metadata.window_lengths or [metadata.window_length]
                )
            }
        )
    references = source_episodes(resolved_source, policy)
    if not references:
        raise ValueError("sequence import requires at least one episode")
    layout = TrajectoryRowLayout(
        policy.observation_columns,
        policy.static_columns,
        policy.actions,
        mask_column=config.mask_column
        or policy.raw_mask_column
        or resolved_source.declaration.mask_column
        or (resolved_source.manifest or {}).get("mask_column"),
        aliases=resolved_source.declaration.action_columns
        or (resolved_source.manifest or {}).get("action_columns"),
    )
    stage = work / "publication"
    stage.mkdir()
    tables, shards, seen = [], [], set()

    names_by_id = {}
    sequence_seeds = {}
    seed_path = work / "sequence-seeds.json"
    if seed_path.is_file():
        sequence_seeds = json.loads(seed_path.read_text())

    def record_shard(path):
        with pq.ParquetFile(path) as file:
            ids = file.read(columns=["sequenceId"]).column(0).unique().to_pylist()
            shards.append(
                {
                    "path": path.name,
                    "rows": file.metadata.num_rows,
                    "sequences": sorted(ids),
                }
            )

    for path in sorted(work.glob("part-*.parquet")):
        coordinates = pq.read_table(path, columns=["sequenceId", SEQUENCE_NAME_COLUMN])
        for row in (
            coordinates.group_by(["sequenceId", SEQUENCE_NAME_COLUMN])
            .aggregate([])
            .to_pylist()
        ):
            name = row[SEQUENCE_NAME_COLUMN]
            register_sequence_name(names_by_id, row["sequenceId"], name)
            seen.add(name)
        record_shard(path)

    def flush():
        if tables:
            shard_schema = pa.unify_schemas(
                [table.schema for table in tables], promote_options="permissive"
            )
            table = pa.concat_tables(
                [cast_episode_table(table, shard_schema) for table in tables]
            )
            path = work / f"part-{len(shards):05d}.parquet"
            with atomic_path(seed_path) as temporary:
                temporary.write_text(json.dumps(sequence_seeds))
            with atomic_path(path) as temporary:
                pq.write_table(table, temporary)
            record_shard(path)
            tables.clear()

    for index in range(len(seen), len(references)):
        episode = references[index].load()
        if episode.episode_id in seen:
            raise ValueError("duplicate episode IDs in sequence import")
        seen.add(episode.episode_id)
        seq = episode.sequence_id if episode.sequence_id is not None else index
        register_sequence_name(names_by_id, seq, episode.episode_id)
        context = episode.context
        if context:
            sequence_seeds[str(seq)] = seed_metadata(context)
        table = episode_table(seq, episode, layout, policy, first_position=0)
        passthrough = [
            name
            for name in config.passthrough_columns
            if name != "__collection" or name not in table.column_names
        ]
        if passthrough:
            reference = references[index]
            reserved = (
                set(table.column_names)
                | {ACTION_METADATA}
                | set(policy.actions)
                | set(policy.feedback)
            )
            if overlap := reserved.intersection(passthrough):
                raise ValueError(
                    f"Passthrough columns overlap sequence fields: {sorted(overlap)}"
                )
            auxiliary = reference.source_columns(
                range(episode.first_position, episode.first_position + len(table)),
                passthrough,
            ).to_arrow()
            for name in auxiliary.column_names:
                column = auxiliary[name]
                table = table.append_column(name, column)
        tables.append(table)
        if len(tables) == config.episodes_per_shard:
            flush()
    flush()
    publish_sequences(
        stage / "manifest.json",
        shards,
        policy.observation_columns,
        {
            **resolved_source.provenance(),
            "policy_binding": policy.task_binding.model_dump(mode="json"),
            **({"sequence_seeds": sequence_seeds} if sequence_seeds else {}),
        },
        static=policy.static_columns,
        layout=layout,
        source_root=work,
    )
    os.rename(stage, destination)
