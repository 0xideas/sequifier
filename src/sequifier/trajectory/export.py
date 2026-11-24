"""Atomically export canonical episodes as checksummed raw Parquet shards."""

import os
import sys
from pathlib import Path

import pyarrow as pa

from sequifier.io.action_record import action_record_layout
from sequifier.io.atomic import atomic_path
from sequifier.io.preprocessing_workspace import preprocessing_workspace
from sequifier.io.sequence import (
    CONTEXT_COLUMN,
    CONTEXT_RECORDS,
    TransitionColumns,
    register_sequence_name,
)
from sequifier.trajectory.importers import source_episodes
from sequifier.trajectory.sequence_output import publish_sequences
from sequifier.trajectory.tables import (
    ACTION_RECORD,
    GROUP_COLUMN,
    PREVIOUS_ACTION_RECORD,
    REWARD_COLUMN,
    SEQUENCE_NAME_COLUMN,
    SUCCESSOR_PREFIX,
    TRANSITION_FLAGS,
)

MAX_FLOAT64_INTEGER = int(sys.float_info.max)


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _float64_value(value, dtype, column):
    if value is None:
        return value
    if pa.types.is_float64(dtype) and _is_integer(value):
        if abs(value) > MAX_FLOAT64_INTEGER:
            raise ValueError(f"{column}: integer exceeds the finite float64 range")
        converted = float(value)
        if converted != value:
            raise ValueError(
                f"{column}: integer cannot be represented exactly as float64"
            )
        return converted
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return [_float64_value(item, dtype.value_type, column) for item in value]
    if pa.types.is_struct(dtype):
        return {
            name: _float64_value(item, dtype.field(name).type, f"{column}.{name}")
            for name, item in value.items()
        }
    return value


def _checked_array(entries, dtype, column):
    dtype = dtype if dtype is not None else pa.infer_type(entries)
    return pa.array(
        [_float64_value(value, dtype, column) for value in entries], type=dtype
    )


def cast_episode_table(table, schema):
    if table.schema == schema:
        return table
    arrays = []
    for field in schema:
        if field.name == CONTEXT_COLUMN and field.name not in table.column_names:
            arrays.append(pa.nulls(len(table), type=field.type))
            continue
        column = table[field.name]
        arrays.append(
            column
            if column.type == field.type
            else _checked_array(column.to_pylist(), field.type, field.name)
        )
    return pa.Table.from_arrays(arrays, schema=schema)


def _action_type(field):
    if field.type == "real":
        return pa.float64()
    if not field.values:
        raise ValueError("categorical sequence actions require declared values")
    if all(isinstance(label, str) for label in field.values):
        return pa.string()
    if any(isinstance(label, str) for label in field.values):
        raise ValueError(
            "categorical sequence actions cannot mix string and numeric labels"
        )
    if all(_is_integer(label) for label in field.values):
        if min(field.values) >= 0 and max(field.values) >= 2**63:
            return pa.uint64()
        return pa.int64()
    return pa.float64()


class EpisodeTableEncoder:
    """Canonical Arrow columns shared by sequence export and policy preparation."""

    def __init__(self, policy):
        fields = policy.observation_columns + policy.static_columns
        successor_columns = [
            SUCCESSOR_PREFIX + name for name in policy.observation_columns
        ]
        reserved = {
            "sequenceId",
            "itemPosition",
            SEQUENCE_NAME_COLUMN,
            ACTION_RECORD,
            PREVIOUS_ACTION_RECORD,
            GROUP_COLUMN,
            REWARD_COLUMN,
            CONTEXT_COLUMN,
            *TRANSITION_FLAGS,
            *successor_columns,
        }
        if set(fields) & reserved:
            raise ValueError(
                "policy feature names collide with sequence import metadata columns"
            )
        auxiliary_types = {"latent": pa.float64(), "component": pa.int64()}
        action_type = pa.struct(
            [
                pa.field(
                    field.key,
                    auxiliary_types[field.kind]
                    if field.kind in auxiliary_types
                    else _action_type(policy.actions[field.action]),
                )
                for field in action_record_layout(policy).fields
            ]
        )
        self.types = {
            REWARD_COLUMN: pa.float64(),
            **{name: pa.bool_() for name in TRANSITION_FLAGS},
            ACTION_RECORD: action_type,
            PREVIOUS_ACTION_RECORD: action_type,
            GROUP_COLUMN: pa.string(),
        }
        self.observations = policy.observation_columns
        self.successors = successor_columns

    def table(self, index, episode):
        length = len(episode.actions)
        transitions = episode.transitions or TransitionColumns()
        values = {
            **episode.observations,
            **{name: [value] * length for name, value in episode.static_inputs.items()},
            "sequenceId": [
                episode.sequence_id if episode.sequence_id is not None else index
            ]
            * length,
            "itemPosition": episode.item_positions,
            SEQUENCE_NAME_COLUMN: [episode.episode_id] * length,
            ACTION_RECORD: episode.actions,
            PREVIOUS_ACTION_RECORD: (
                episode.previous_action,
                *episode.actions[:-1],
            ),
            GROUP_COLUMN: [episode.group_id] * length,
            REWARD_COLUMN: transitions.rewards or [None] * length,
            "terminated": transitions.terminated or [None] * length,
            "truncated": transitions.truncated or [None] * length,
            "bootstrap_allowed": transitions.bootstrap_allowed or [None] * length,
        }
        if context := getattr(episode, "context", None):
            values[CONTEXT_COLUMN] = [
                CONTEXT_RECORDS.dump_json(context).decode(),
                *([None] * (length - 1)),
            ]
        episode_types = dict(self.types)
        for name, column in zip(self.observations, self.successors, strict=True):
            values[column] = [
                successor[name] if successor is not None else None
                for successor in (transitions.successors or [None] * length)
            ]
            dtype = pa.infer_type([*values[name], *values[column]])
            episode_types[name] = episode_types[column] = dtype
        arrays = {
            name: _checked_array(entries, episode_types.get(name), name)
            for name, entries in values.items()
        }
        return pa.table(arrays)


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
    encoder = EpisodeTableEncoder(policy)
    stage = work / "publication"
    stage.mkdir()
    tables, shards, seen = [], [], set()

    names_by_id = {}

    def record_shard(path):
        with pq.ParquetFile(path) as file:
            ids = file.read(columns=["sequenceId"]).column(0).unique().to_pylist()
            shards.append(
                {
                    "path": path.name,
                    "decisions": file.metadata.num_rows,
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
        tables.append(encoder.table(index, episode))
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
        },
        static=policy.static_columns,
        source_root=work,
    )
    os.rename(stage, destination)
