"""Publish domain-valued rollout decisions for sequence preprocessing."""

from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from itertools import islice
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import polars as pl
import pyarrow.parquet as pq

from sequifier.config.column_names import INCOMING_ACTION_PRESENT
from sequifier.contract_versions import ContractVersions
from sequifier.fingerprints import file_checksum, fingerprint
from sequifier.io.action_record import action_record_layout
from sequifier.io.actions import decode_action
from sequifier.io.atomic import atomic_path, write_json_atomic
from sequifier.trajectory.row_layout import ACTION_METADATA, TrajectoryRowLayout
from sequifier.trajectory.table_schema import cast_episode_table, sequence_table

DestinationScope = Callable[[Path], AbstractContextManager[str | Path]]


def sequence_coordinate(identity):
    # Signed int64 is the coordinate type used by both dataset loaders.
    return int(fingerprint(identity)[:15], 16)


def action_record(encoded, policy):
    if encoded is None:
        return None
    layout = action_record_layout(policy)
    return layout.pack(
        decode_action(encoded, policy), *layout.unpack_auxiliary(encoded)
    )


def trajectory_coordinates(identity, position, group):
    return {
        "sequenceId": sequence_coordinate(identity),
        "sequenceName": identity,
        "itemPosition": position,
        "groupId": group,
    }


def trajectory_policy(trajectory):
    from sequifier.config.policy_config import ResolvedPolicyBinding

    binding = trajectory.metadata.get("policy_execution", {}).get("config")
    return ResolvedPolicyBinding.model_validate(binding) if binding else None


def trajectory_layout(trajectory, output=None, mask_column=None):
    return TrajectoryRowLayout(
        trajectory.observation_columns,
        trajectory.seed.static_inputs,
        trajectory.prediction_columns,
        mask_column=mask_column or (output.mask_column if output else None),
        aliases=output.action_columns if output else None,
    )


def trajectory_rows(trajectory, layout, policy=None, previous=None):
    metadata = action_record_layout(policy) if policy else None
    seed = trajectory.seed
    initial = layout.row(
        trajectory_coordinates(trajectory.episode_id, 0, trajectory.seed_group),
        {name: seed.observations[name][-1] for name in trajectory.observation_columns},
        seed.static_inputs,
        action=previous,
        masked=seed.observation_mask[-1],
    )
    if policy:
        initial["__collection"] = None
    yield initial
    for position, arrival in enumerate(trajectory.arrivals, 1):
        action = (
            metadata.pack(
                arrival.action, arrival.action_latents, arrival.action_components
            )
            if metadata
            else arrival.action
        )
        row = layout.row(
            trajectory_coordinates(
                trajectory.episode_id, position, trajectory.seed_group
            ),
            {
                name: arrival.observation[name]
                for name in trajectory.observation_columns
            },
            seed.static_inputs,
            action=action,
            reward=arrival.reward,
            terminated=arrival.terminated,
            truncated=arrival.truncated,
        )
        if policy:
            row["__collection"] = arrival.diagnostics or None
        yield row


def _store_parquet(source, manifest, destination_scope: DestinationScope = nullcontext):
    checksum = file_checksum(source)
    parts = manifest.with_name(manifest.name + ".parts")
    destination = parts / (checksum + ".parquet")
    with destination_scope(destination) as staged:
        with atomic_path(staged, durable=True) as temporary:
            copyfile(source, temporary)
    return {"path": str(destination.relative_to(manifest.parent)), "sha256": checksum}


def arrival_table(batch, layout, policy, collection_type=None):
    columns = {key: [row[key] for row in batch] for key in batch[0]}
    if not any(columns[ACTION_METADATA]):
        del columns[ACTION_METADATA]
    return sequence_table(
        columns, layout, policy=policy, collection_type=collection_type
    )


def write_parts(manifest, rows, *, layout, policy=None, collection_type=None):
    """Bound memory per batch and retain immutable training checkpoint prefixes."""
    manifest = Path(manifest)
    rows = iter(rows)
    with TemporaryDirectory() as work:
        shard = Path(work) / "batch.parquet"
        while batch := list(islice(rows, 64)):
            table = arrival_table(batch, layout, policy, collection_type)
            pq.write_table(table, shard, compression="zstd")
            yield {
                **_store_parquet(shard, manifest),
                "rows": len(table),
                "sequences": sorted(table["sequenceId"].unique().to_pylist()),
            }


def validate_arrivals(scan, actions, observations=()):
    """Check a complete dataset, including sequences spanning multiple parts."""
    pos = pl.col("itemPosition")
    schema = scan.collect_schema()
    for name in ("itemPosition", "sequenceId"):
        if not schema[name].is_integer():
            raise ValueError(f"{name} must contain integers")
    present = pl.col(list(actions)).is_not_null()
    complete = pl.all_horizontal(present)
    outcomes = [
        name for name in ("reward", "terminated", "truncated") if name in schema
    ]
    boundaries = [name for name in ("terminated", "truncated") if name in schema]
    invalid = ((pos > 0) | pl.any_horizontal(present)) & ~complete
    if observations:
        invalid |= pl.any_horizontal(pl.col(list(observations)).is_null())
    if outcomes:
        invalid |= (pos == 0) & pl.any_horizontal(pl.col(outcomes).is_not_null())
    bounds = (
        scan.group_by("sequenceId")
        .agg(
            pl.len().alias("rows"),
            pos.n_unique().alias("unique"),
            pos.min().alias("first"),
            pos.max().alias("last"),
            (invalid | pos.is_null()).any().alias("invalid"),
            (
                pos.filter(pl.any_horizontal(pl.col(boundaries).fill_null(False))).min()
                if boundaries
                else pl.lit(None)
            ).alias("boundary"),
        )
        .collect()
    )
    if (
        bounds.is_empty()
        or bounds.filter(
            pl.col("sequenceId").is_null()
            | pl.col("invalid")
            | (pl.col("rows") < 2)
            | (pl.col("unique") != pl.col("rows"))
            | (pl.col("first") != 0)
            | (pl.col("last") + 1 != pl.col("rows"))
            | (pl.col("boundary") < pl.col("last"))
        ).height
    ):
        raise ValueError(
            "Expected contiguous observation rows from zero, with complete incoming actions and a final observation"
        )
    return bounds


def publish_sequences(
    manifest,
    parts,
    observations,
    provenance,
    *,
    static=(),
    layout,
    pin=False,
    source_root=None,
    destination_scope: DestinationScope = nullcontext,
):
    """Publish one contiguous trajectory table and manifest for every producer."""
    import pyarrow as pa

    manifest = Path(manifest)
    root = manifest.parent if source_root is None else Path(source_root)
    if not parts:
        raise ValueError("A rollout dataset requires at least one decision")
    paths = [root / part["path"] for part in parts]
    with TemporaryDirectory() as work:
        compact = Path(work) / "trajectory.parquet"
        schema = pa.unify_schemas(
            [pq.read_schema(path) for path in paths], promote_options="permissive"
        )
        staged = Path(work) / "parts.parquet"
        with pq.ParquetWriter(staged, schema, compression="zstd") as writer:
            for path in paths:
                with pq.ParquetFile(path) as source:
                    for batch in source.iter_batches():
                        writer.write_table(
                            cast_episode_table(pa.Table.from_batches([batch]), schema)
                        )
        scan = pl.scan_parquet(staged)
        bounds = validate_arrivals(scan, layout.actions.values(), observations)
        fields = ["sequenceName", "groupId", *static]
        metadata_path = Path(work) / "sequences.parquet"
        metadata = scan.select("sequenceId", *fields).unique()
        if metadata.select(pl.len()).collect().item() != len(bounds):
            raise ValueError(
                "Sequence identity, group and static inputs must be constant"
            )
        metadata.sort("sequenceId").sink_parquet(metadata_path)
        sequence_metadata = _store_parquet(metadata_path, manifest, destination_scope)
        scan.drop(fields).sort(["sequenceId", "itemPosition"]).sink_parquet(
            compact, compression="zstd"
        )
        shard = {
            **_store_parquet(compact, manifest, destination_scope),
            "decisions": int(bounds["rows"].sum() - len(bounds)),
            "sequences": sorted(bounds["sequenceId"].to_list()),
        }
    values = {
        "format": "sequifier.sequence-data",
        "version": ContractVersions.SEQUENCE_DATA_MANIFEST,
        **layout.manifest_fields(),
        "observation_columns": list(observations),
        "static_columns": list(static),
        "episodes": len(bounds),
        "decisions": shard["decisions"],
        "provenance": provenance,
        "shards": [shard],
        "sequence_metadata": sequence_metadata,
    }
    reports = {}
    if pin:
        data_file = manifest.parent / shard["path"]
        pinned = {**values, "shards": [{**shard, "path": data_file.name}]}
        pinned["sequence_metadata"] = {
            **sequence_metadata,
            "path": Path(sequence_metadata["path"]).name,
        }
        snapshot = data_file.parent / (fingerprint(pinned) + ".manifest.json")
        write_json_atomic(snapshot, pinned)
        reports = {
            "committed_sequence_manifest": str(snapshot),
            "committed_sequence_manifest_sha256": file_checksum(snapshot),
        }
    with destination_scope(manifest) as staged:
        write_json_atomic(staged, values)
        return {
            **reports,
            "sequence_manifest": str(manifest),
            "sequence_manifest_sha256": file_checksum(staged),
        }


def publish_inference_rollout(trajectory, config):
    from sequifier.helpers import normalize_path
    from sequifier.io.inference_workspace import inference_output

    rollout = config.rollout
    manifest = normalize_path(rollout.output.manifest_path, config.project_root)
    config.validate_output_destination(manifest, label="sequence manifest")
    config.validate_output_destination(manifest + ".parts", label="sequence shards")
    with TemporaryDirectory() as work:
        policy = trajectory_policy(trajectory)
        layout = trajectory_layout(
            trajectory,
            rollout.output,
            (
                policy.raw_mask_column
                if policy and not rollout.output.mask_column
                else None
            ),
        )
        from sequifier.io.sequence import seed_metadata
        from sequifier.trajectory.importers import trajectory_seed_context

        context, previous = trajectory_seed_context(
            trajectory.seed, trajectory.metadata.get("policy_execution", {})
        )
        parts = list(
            write_parts(
                Path(work) / "manifest.json",
                trajectory_rows(trajectory, layout, policy, previous),
                policy=policy,
                layout=layout,
            )
        )
        publish_sequences(
            manifest,
            parts,
            trajectory.observation_columns,
            {
                "task": config.task,
                **(
                    {"policy_binding": policy.task_binding.model_dump(mode="json")}
                    if policy
                    else {}
                ),
                "bindings": [b.model_dump() for b in trajectory.bindings],
                "seed_origin": trajectory.metadata["seed_origin"],
                "model_path": config.model_paths[0],
                "collection_cutoff": trajectory.collection_cutoff,
                "sequence_seeds": {
                    str(sequence_coordinate(trajectory.episode_id)): seed_metadata(
                        context
                    )
                }
                if context
                else {},
            },
            static=trajectory.seed.static_inputs,
            layout=layout,
            source_root=work,
            destination_scope=inference_output,
        )


@contextmanager
def supervised_manifest_input(
    path,
    *,
    dependencies=None,
    mask_column=None,
    split_column=None,
    curriculum_columns=(),
):
    """Feed the verified shards to ordinary file preprocessing, including split fitting."""
    from sequifier.config.policy_train_config import SequenceSourceConfig
    from sequifier.trajectory.sources import resolve_source
    from sequifier.trajectory.tables import observation_mask, scan_tables

    path = Path(path).resolve()
    if dependencies is not None:
        dependencies.add(path)
    with TemporaryDirectory() as work:
        name = path.parent.name if path.stem == "manifest" else path.stem
        table = Path(work) / (name + ".parquet")
        resolved = resolve_source(
            SequenceSourceConfig(manifest_path=str(path)), path.parent
        )
        manifest = resolved.manifest
        assert manifest is not None and resolved.static_path is not None
        if dependencies is not None:
            dependencies.merge(resolved.dependencies)
        scan = scan_tables(resolved.paths)
        action_fields = list(manifest["action_columns"].values())
        validate_arrivals(scan, action_fields, manifest["observation_columns"])
        scan = scan.with_columns(
            pl.col(action_fields[0])
            .is_not_null()
            .cast(pl.Float32)
            .alias(INCOMING_ACTION_PRESENT)
        )
        static = manifest["static_columns"]
        static_path = None
        if static:
            static_path = table.with_suffix(".static.parquet")
            pl.scan_parquet(resolved.static_path).select(
                "sequenceId", *static
            ).sink_parquet(static_path)
        auxiliary = set(curriculum_columns) | {split_column}
        selected = [
            "sequenceName",
            "groupId",
            *(field for field in static if field in auxiliary),
        ]
        if selected:
            scan = scan.join(
                pl.scan_parquet(resolved.static_path).select("sequenceId", *selected),
                on="sequenceId",
                validate="m:1",
            )
        stored_mask = manifest["mask_column"]
        mask = observation_mask(scan, mask_column, recorded_column=stored_mask)
        target_mask = mask_column
        if target_mask is None and scan.select(mask.any()).collect().item():
            target_mask = stored_mask
        if target_mask:
            scan = scan.with_columns(mask.alias(target_mask))
        if target_mask != stored_mask:
            scan = scan.drop(stored_mask)
        manifest = {**manifest, "mask_column": target_mask}
        scan.sink_parquet(table)
        resolved.dependencies.revalidate()
        yield table, manifest, static_path
