"""Atomically export canonical episodes as checksummed raw Parquet shards."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from sequifier.contract_versions import ContractVersions
from sequifier.io.actions import action_component_key, action_latent_key
from sequifier.io.sequence import TransitionColumns
from sequifier.trajectory.importers import source_episodes
from sequifier.trajectory.sources import file_checksum, source_paths
from sequifier.trajectory.tables import (
    ACTION_RECORD,
    EPISODE_ID_COLUMN,
    GROUP_COLUMN,
    PREVIOUS_ACTION_RECORD,
    REWARD_COLUMN,
    SUCCESSOR_PREFIX,
    TRANSITION_FLAGS,
)


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
    references = source_episodes(source, policy)
    if not references:
        raise ValueError("sequence import requires at least one episode")
    action_type = pa.struct(
        [
            pa.field(
                name,
                pa.string()
                if field.type == "categorical" and isinstance(field.values[0], str)
                else pa.float64(),
            )
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
            table = pa.concat_tables(tables, promote_options="permissive")
            schema = schema or table.schema
            name = f"part-{len(shards):05d}.parquet"
            pq.write_table(table.cast(schema), stage / name)
            shards.append(
                {
                    "path": name,
                    "sha256": file_checksum(stage / name),
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
                **{
                    field: getattr(transitions, field) or [None] * length
                    for field in TRANSITION_FLAGS
                },
            }
            arrays = {
                name: pa.array(entries, type=types.get(name))
                for name, entries in values.items()
            }
            for name, column in zip(
                policy.observation_columns, successor_columns, strict=True
            ):
                arrays[column] = pa.array(
                    [
                        successor[name] if successor is not None else None
                        for successor in (transitions.successors or [None] * length)
                    ],
                    type=arrays[name].type,
                )
            tables.append(pa.table(arrays))
            if (index + 1) % config.episodes_per_shard == 0:
                flush()
        flush()
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
                    {"path": str(path), "sha256": file_checksum(path)}
                    for path in source_paths(source)
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
