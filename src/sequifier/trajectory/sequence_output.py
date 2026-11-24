"""Publish domain-valued rollout decisions for sequence preprocessing."""

import json
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from itertools import islice
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from typing import cast

import polars as pl
import pyarrow.parquet as pq

from sequifier.contract_versions import ContractVersions
from sequifier.fingerprints import file_checksum, fingerprint
from sequifier.io.action_record import action_record_layout
from sequifier.io.actions import decode_action
from sequifier.io.atomic import atomic_path, write_json_atomic
from sequifier.io.sequence import (
    CONTEXT_COLUMN,
    CONTEXT_RECORDS,
    OBSERVATION_MASK,
    ContextObservation,
)
from sequifier.io.values import portable_value
from sequifier.trajectory.table_schema import cast_episode_table, sequence_table
from sequifier.trajectory.tables import ACTION_RECORD, PREVIOUS_ACTION_RECORD

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


def decision_row(
    identity,
    step,
    observation,
    static,
    *,
    action,
    previous,
    reward,
    successor,
    terminated,
    truncated,
    group=None,
    predictions=None,
    masked=False,
):
    row = {
        "sequenceId": sequence_coordinate(identity),
        "sequenceName": identity,
        "itemPosition": step,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "bootstrap_allowed": not terminated,
        "__group": group,
        CONTEXT_COLUMN: None,
        OBSERVATION_MASK: masked,
        **{
            "__next_" + key: None if successor is None else successor[key]
            for key in observation
        },
    }
    if predictions is not None:
        row.update({"__prediction_" + key: value for key, value in predictions.items()})
    if action is not None:
        row[ACTION_RECORD] = action
        row[PREVIOUS_ACTION_RECORD] = previous
    if (
        row.keys() & (observation.keys() | static.keys())
        or observation.keys() & static.keys()
    ):
        raise ValueError(
            "Rollout features overlap sequence coordinates or transition fields"
        )
    row.update(observation)
    row.update(static)
    return {name: portable_value(value) for name, value in row.items()}


def trajectory_policy(trajectory):
    from sequifier.config.policy_config import ResolvedPolicyBinding

    binding = trajectory.metadata.get("policy_execution", {}).get("config")
    return ResolvedPolicyBinding.model_validate(binding) if binding else None


def trajectory_rows(trajectory, policy=None):
    from sequifier.trajectory.importers import sequence_from_trajectory

    policy = policy or trajectory_policy(trajectory)
    episode = sequence_from_trajectory(trajectory, policy) if policy else None
    group = trajectory.seed_group
    scalar_policy = trajectory.format in {
        "sequifier.scalar-policy-trajectory",
        "sequifier.policy-trajectory",
    }
    for i, step in enumerate(trajectory.transitions):
        row = decision_row(
            trajectory.episode_id,
            i,
            step.observation,
            trajectory.seed.static_inputs,
            action=episode.actions[i]
            if episode
            else (step.predictions if scalar_policy else None),
            previous=(episode.actions[i - 1] if i else episode.previous_action)
            if episode
            else (
                trajectory.transitions[i - 1].predictions
                if i and scalar_policy
                else None
            ),
            predictions=step.predictions,
            masked=i == 0 and trajectory.seed.observation_mask[-1],
            reward=step.reward,
            successor=step.successor_observation,
            terminated=step.terminated,
            truncated=step.truncated,
            group=group,
        )
        if i == 0 and (context_steps := getattr(episode, "context", None)):
            row[CONTEXT_COLUMN] = CONTEXT_RECORDS.dump_json(context_steps).decode()
        elif i == 0 and not episode:
            context = [
                ContextObservation(
                    masked=trajectory.seed.observation_mask[index],
                    observation={
                        name: values[index]
                        for name, values in trajectory.seed.observations.items()
                    },
                )
                for index, valid in enumerate(trajectory.seed.attention_valid_mask[:-1])
                if valid
            ]
            if context:
                row[CONTEXT_COLUMN] = CONTEXT_RECORDS.dump_json(tuple(context)).decode()
        yield row


def _store_parquet(source, manifest, destination_scope: DestinationScope = nullcontext):
    checksum = file_checksum(source)
    parts = manifest.with_name(manifest.name + ".parts")
    destination = parts / (checksum + ".parquet")
    with destination_scope(destination) as staged:
        with atomic_path(staged, durable=True) as temporary:
            copyfile(source, temporary)
    return {"path": str(destination.relative_to(manifest.parent)), "sha256": checksum}


def _decision_table(batch, observations, policy, collection_type=None):
    columns = {key: [row[key] for row in batch] for key in batch[0]}
    return sequence_table(
        columns, observations, policy=policy, collection_type=collection_type
    )


def write_parts(manifest, rows, *, observations, policy=None, collection_type=None):
    """Bound memory per batch and retain immutable training checkpoint prefixes."""
    manifest = Path(manifest)
    rows = iter(rows)
    with TemporaryDirectory() as work:
        shard = Path(work) / "batch.parquet"
        while batch := list(islice(rows, 64)):
            table = _decision_table(batch, observations, policy, collection_type)
            pq.write_table(table, shard, compression="zstd")
            yield {
                **_store_parquet(shard, manifest),
                "decisions": len(table),
                "sequences": sorted(table["sequenceId"].unique().to_pylist()),
            }


def publish_sequences(
    manifest,
    parts,
    observations,
    provenance,
    *,
    static=(),
    pin=False,
    source_root=None,
    destination_scope: DestinationScope = nullcontext,
):
    """Publish the same single decision table and manifest for every rollout producer."""
    import pyarrow as pa

    manifest = Path(manifest)
    root = manifest.parent if source_root is None else Path(source_root)
    if not parts:
        raise ValueError("A rollout dataset requires at least one decision")
    if len(parts) == 1 and root == manifest.parent and destination_scope is nullcontext:
        shard = parts[0]
    else:
        paths = [root / part["path"] for part in parts]
        with TemporaryDirectory() as work:
            compact = Path(work) / "decisions.parquet"
            if len(paths) == 1:
                compact = paths[0]
            else:
                schema = pa.unify_schemas(
                    [pq.read_schema(path) for path in paths],
                    promote_options="permissive",
                )
                with pq.ParquetWriter(compact, schema, compression="zstd") as writer:
                    for path in paths:
                        with pq.ParquetFile(path) as source:
                            writer.write_table(
                                cast_episode_table(source.read(), schema)
                            )
            shard = {
                **_store_parquet(compact, manifest, destination_scope),
                "decisions": sum(part["decisions"] for part in parts),
                "sequences": sorted(
                    {seq for part in parts for seq in part["sequences"]}
                ),
            }
    values = manifest_values([shard], observations, provenance, static=static)
    reports = {}
    if pin:
        data_file = manifest.parent / shard["path"]
        pinned = {**values, "shards": [{**shard, "path": data_file.name}]}
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


def manifest_values(parts, observations, provenance, *, static=()):
    return {
        "format": "sequifier.sequence-data",
        "version": ContractVersions.SEQUENCE_DATA_MANIFEST,
        "observation_columns": list(observations),
        "static_columns": list(static),
        "episodes": len({seq for part in parts for seq in part["sequences"]}),
        "decisions": sum(p["decisions"] for p in parts),
        "provenance": provenance,
        "shards": parts,
    }


def publish_inference_rollout(trajectory, config):
    from sequifier.helpers import normalize_path
    from sequifier.io.inference_workspace import (
        inference_output,
        write_inference_trajectory,
    )

    rollout = config.rollout
    if rollout.trajectory_path:
        destination = normalize_path(rollout.trajectory_path, config.project_root)
        config.validate_output_destination(destination, label="trajectory_path")
        write_inference_trajectory(trajectory, destination)
    if rollout.output:
        manifest = normalize_path(rollout.output.manifest_path, config.project_root)
        config.validate_output_destination(manifest, label="sequence manifest")
        config.validate_output_destination(manifest + ".parts", label="sequence shards")
        with TemporaryDirectory() as work:
            policy = trajectory_policy(trajectory)
            parts = list(
                write_parts(
                    Path(work) / "manifest.json",
                    trajectory_rows(trajectory, policy),
                    policy=policy,
                    observations=trajectory.seed.observations,
                )
            )
            publish_sequences(
                manifest,
                parts,
                trajectory.seed.observations,
                {
                    "task": config.task,
                    "bindings": [b.model_dump() for b in trajectory.bindings],
                    "seed_origin": trajectory.metadata["seed_origin"],
                    "model_path": config.model_paths[0],
                    "prediction_columns": trajectory.prediction_columns,
                    "collection_cutoff": trajectory.collection_cutoff,
                    "seed_history": trajectory.seed.model_dump(),
                },
                static=trajectory.seed.static_inputs,
                source_root=work,
                destination_scope=inference_output,
            )


@contextmanager
def supervised_manifest_input(
    path, sequence_view="decisions", *, dependencies=None, mask_column=None
):
    """Feed the verified shards to ordinary file preprocessing, including split fitting."""
    from sequifier.config.policy_train_config import SequenceSourceConfig
    from sequifier.trajectory.sources import resolve_source
    from sequifier.trajectory.tables import scan_tables

    path = Path(path).resolve()
    if dependencies is not None:
        dependencies.add(path)
    with TemporaryDirectory() as work:
        name = path.parent.name if path.stem == "manifest" else path.stem
        table = Path(work) / (name + ".parquet")
        with path.open() as stream:
            kind = json.load(stream).get("format")
        resolved = None
        if kind in {
            "sequifier.numeric-forecast-trajectory",
            "sequifier.policy-trajectory",
            "sequifier.scalar-policy-trajectory",
        }:
            from sequifier.trajectory.store import load_trajectory

            trajectory = load_trajectory(path)
            policy = trajectory_policy(trajectory)
            scan = cast(
                pl.DataFrame,
                pl.from_arrow(
                    _decision_table(
                        list(trajectory_rows(trajectory, policy)),
                        trajectory.seed.observations,
                        policy,
                    )
                ),
            ).lazy()
            manifest = {
                "observation_columns": list(trajectory.seed.observations),
                "static_columns": list(trajectory.seed.static_inputs),
                "provenance": {
                    "task": "forecast"
                    if kind == "sequifier.numeric-forecast-trajectory"
                    else "policy"
                },
                "shards": [],
            }
        else:
            resolved = resolve_source(
                SequenceSourceConfig(manifest_path=str(path)), path.parent
            )
            manifest = resolved.manifest
            if resolved.kind != "table" or manifest is None:
                raise ValueError(
                    "Supervised sequence manifests require raw table shards"
                )
            if dependencies is not None:
                dependencies.merge(resolved.dependencies)
            scan = scan_tables(resolved.paths)
        if OBSERVATION_MASK not in scan.collect_schema():
            scan = scan.with_columns(pl.lit(False).alias(OBSERVATION_MASK))
        if ACTION_RECORD in scan.collect_schema() and sequence_view == "decisions":
            dtype = scan.collect_schema()[ACTION_RECORD]
            if isinstance(dtype, pl.Struct):
                columns = scan.collect_schema().names()
                projected = []
                for field in dtype.fields:
                    name = (
                        field.name
                        if field.name not in columns
                        else "__executed_" + field.name
                    )
                    if name in columns:
                        raise ValueError(
                            f"Executed action projection overlaps existing column {name!r}"
                        )
                    projected.append(
                        pl.col(ACTION_RECORD).struct.field(field.name).alias(name)
                    )
                scan = scan.with_columns(projected)
        if sequence_view == "observations":
            observations = manifest.get("observation_columns")
            if not observations:
                raise ValueError(
                    "Observation view requires declared observation columns"
                )
            # The successor of the final decision is a real observation even when
            # no subsequent decision exists. Add it exactly once across all shards.
            last = scan.sort("itemPosition").group_by("sequenceId").last()
            final = last.filter(
                pl.all_horizontal(
                    [pl.col("__next_" + c).is_not_null() for c in observations]
                )
            )
            coordinates = ["sequenceId", "itemPosition", "sequenceName", "__group"]
            static = manifest.get("static_columns", [])
            columns = coordinates + observations + list(static) + [OBSERVATION_MASK]
            final = final.with_columns(
                pl.lit(False).alias(OBSERVATION_MASK),
                (pl.col("itemPosition") + 1).alias("itemPosition"),
                *[pl.col("__next_" + c).alias(c) for c in observations],
            )
            scan = pl.concat(
                [scan.select(columns), final.select(columns)], how="vertical_relaxed"
            )
        static = manifest.get("static_columns", [])
        static_path = None
        if static:
            static_path = table.with_suffix(".static.parquet")
            scan.select("sequenceId", *static).unique().sink_parquet(static_path)
            scan = scan.drop(static)
        if mask_column and mask_column != OBSERVATION_MASK:
            if mask_column in scan.collect_schema():
                scan = scan.with_columns(
                    (
                        pl.col(mask_column).cast(pl.Boolean) | pl.col(OBSERVATION_MASK)
                    ).alias(mask_column)
                ).drop(OBSERVATION_MASK)
            else:
                scan = scan.rename({OBSERVATION_MASK: mask_column})
        elif not mask_column:
            manifest = {
                **manifest,
                "masked_observations": scan.select(pl.col(OBSERVATION_MASK).any())
                .collect()
                .item(),
            }
        scan.sink_parquet(table)
        if resolved is not None:
            resolved.dependencies.revalidate()
        yield table, manifest, static_path
