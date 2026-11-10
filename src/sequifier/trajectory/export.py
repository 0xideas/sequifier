"""Atomically export canonical episodes as checksummed raw Parquet shards."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from sequifier.config.policy_train_config import PolicyTableColumns
from sequifier.io.actions import action_component_key, action_latent_key
from sequifier.io.sequence import TransitionColumns
from sequifier.trajectory.importers import source_episodes
from sequifier.trajectory.sources import file_checksum, source_paths


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
    columns = PolicyTableColumns(
        action="__executed_action",
        reward="__reward",
        terminated="__terminated",
        truncated="__truncated",
        previous_action="__previous_action",
        bootstrap_allowed="__bootstrap",
        group="__group",
        episode_id="episodeId",
        successors={name: "__next_" + name for name in policy.observation_columns},
    )
    fields = policy.observation_columns + policy.static_columns
    reserved = {
        value for value in columns.model_dump().values() if isinstance(value, str)
    } | set(columns.successors.values())
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
    types = {name: pa.float64() for name in ({"__reward"})}
    types.update(
        {name: pa.bool_() for name in ("__terminated", "__truncated", "__bootstrap")}
    )
    types.update(
        {
            "__executed_action": action_type,
            "__previous_action": action_type,
            "__group": pa.string(),
        }
    )
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
                "episodeId": [episode.episode_id] * length,
                "__executed_action": episode.actions,
                "__previous_action": (episode.previous_action, *episode.actions[:-1]),
                "__group": [episode.group_id] * length,
                **{
                    column: getattr(transitions, field) or [None] * length
                    for field, column in {
                        "rewards": "__reward",
                        "terminated": "__terminated",
                        "truncated": "__truncated",
                        "bootstrap_allowed": "__bootstrap",
                    }.items()
                },
            }
            arrays = {
                name: pa.array(entries, type=types.get(name))
                for name, entries in values.items()
            }
            for name, column in columns.successors.items():
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
            "version": 1,
            "episodes": len(seen),
            "decisions": sum(shard["decisions"] for shard in shards),
            "columns": columns.model_dump(),
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
