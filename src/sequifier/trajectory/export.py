"""Atomically export canonical episodes as checksummed raw Parquet shards."""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from sequifier.contract_versions import ContractVersions
from sequifier.io.actions import action_component_key, action_latent_key
from sequifier.io.sequence import TransitionColumns
from sequifier.trajectory.importers import source_episodes
from sequifier.trajectory.sources import file_checksum, resolve_source
from sequifier.trajectory.tables import (
    ACTION_RECORD,
    EPISODE_ID_COLUMN,
    GROUP_COLUMN,
    PREVIOUS_ACTION_RECORD,
    REWARD_COLUMN,
    SUCCESSOR_PREFIX,
    TRANSITION_FLAGS,
)

MAX_FLOAT64_INTEGER = int(sys.float_info.max)


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _float64_value(value, dtype, column, pa):
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
        return [_float64_value(item, dtype.value_type, column, pa) for item in value]
    if pa.types.is_struct(dtype):
        return {
            name: _float64_value(item, dtype.field(name).type, f"{column}.{name}", pa)
            for name, item in value.items()
        }
    return value


def _checked_array(entries, dtype, column, pa):
    dtype = dtype if dtype is not None else pa.infer_type(entries)
    return pa.array(
        [_float64_value(value, dtype, column, pa) for value in entries], type=dtype
    )


def _cast_table(table, schema, pa):
    arrays = []
    for field in schema:
        column = table[field.name]
        arrays.append(
            column
            if column.type == field.type
            else _checked_array(column.to_pylist(), field.type, field.name, pa)
        )
    return pa.Table.from_arrays(arrays, schema=schema)


def _action_type(field, pa):
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


def import_sequences(config) -> Path:
    """Commit canonical columnar episodes as checksummed raw Parquet shards."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = config.resolved_source()
    policy = config.resolved_policy()
    destination = (Path(config.project_root) / config.output_path).resolve()
    if destination.exists():
        raise ValueError(
            "sequence import output_path must name a new dataset directory"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = policy.observation_columns + policy.static_columns
    successor_columns = [SUCCESSOR_PREFIX + name for name in policy.observation_columns]
    reserved = {
        "sequenceId",
        "itemPosition",
        EPISODE_ID_COLUMN,
        ACTION_RECORD,
        PREVIOUS_ACTION_RECORD,
        GROUP_COLUMN,
        REWARD_COLUMN,
        *TRANSITION_FLAGS,
        *successor_columns,
    }
    if set(fields) & reserved:
        raise ValueError(
            "policy feature names collide with sequence import metadata columns"
        )
    resolved_source = resolve_source(source)
    references = source_episodes(resolved_source, policy)
    if not references:
        raise ValueError("sequence import requires at least one episode")
    action_type = pa.struct(
        [
            pa.field(name, _action_type(field, pa))
            for name, field in policy.actions.items()
        ]
        + [
            pa.field(action_latent_key(name, policy), pa.float64())
            for name, field in policy.actions.items()
            if field.type == "real"
        ]
        + [
            pa.field(action_component_key(name, policy), pa.int64())
            for name, field in policy.actions.items()
            if field.type == "real"
        ]
    )
    types = {
        REWARD_COLUMN: pa.float64(),
        **{name: pa.bool_() for name in TRANSITION_FLAGS},
        ACTION_RECORD: action_type,
        PREVIOUS_ACTION_RECORD: action_type,
        GROUP_COLUMN: pa.string(),
    }
    stage = Path(tempfile.mkdtemp(prefix=".sequifier-import-", dir=destination.parent))
    tables, shards, seen, schema = [], [], set(), None

    def flush():
        nonlocal schema
        if tables:
            shard_schema = pa.unify_schemas(
                [table.schema for table in tables], promote_options="permissive"
            )
            table = pa.concat_tables(
                [_cast_table(table, shard_schema, pa) for table in tables]
            )
            schema = (
                shard_schema
                if schema is None
                else pa.unify_schemas(
                    [schema, shard_schema], promote_options="permissive"
                )
            )
            name = f"part-{len(shards):05d}.parquet"
            pq.write_table(table, stage / name)
            shards.append(
                {
                    "path": name,
                    "decisions": table.num_rows,
                }
            )
            tables.clear()

    try:
        for index, reference in enumerate(references):
            episode = reference.load()
            if episode.episode_id in seen:
                raise ValueError("duplicate episode IDs in sequence import")
            seen.add(episode.episode_id)
            length = len(episode.actions)
            transitions = episode.transitions or TransitionColumns()
            values = {
                **episode.observations,
                **{
                    name: [value] * length
                    for name, value in episode.static_inputs.items()
                },
                "sequenceId": [index] * length,
                "itemPosition": episode.item_positions,
                EPISODE_ID_COLUMN: [episode.episode_id] * length,
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
            episode_types = dict(types)
            for name, column in zip(
                policy.observation_columns, successor_columns, strict=True
            ):
                values[column] = [
                    successor[name] if successor is not None else None
                    for successor in (transitions.successors or [None] * length)
                ]
                dtype = pa.infer_type([*values[name], *values[column]])
                episode_types[name] = episode_types[column] = dtype
            arrays = {
                name: _checked_array(entries, episode_types.get(name), name, pa)
                for name, entries in values.items()
            }
            tables.append(pa.table(arrays))
            if (index + 1) % config.episodes_per_shard == 0:
                flush()
        flush()
        # Promote earlier shards only when a later episode widened their schema.
        for shard in shards:
            path = stage / shard["path"]
            if pq.read_schema(path) != schema:
                table = _cast_table(pq.read_table(path), schema, pa)
                pq.write_table(table, path)
            shard["sha256"] = file_checksum(path)
        manifest = {
            "format": "sequifier.sequence-data",
            "version": ContractVersions.SEQUENCE_DATA_MANIFEST,
            "episodes": len(seen),
            "decisions": sum(shard["decisions"] for shard in shards),
            "policy_schema": policy.model_dump(mode="json"),
            "shards": shards,
            "provenance": {
                "source": source.model_dump(mode="json"),
                "files": [
                    {"path": str(path.resolve()), "sha256": checksum}
                    for path, checksum in sorted(resolved_source.checksums.items())
                ],
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.rename(stage, destination)
        return destination / "manifest.json"
    finally:
        if stage.exists():
            shutil.rmtree(stage)
