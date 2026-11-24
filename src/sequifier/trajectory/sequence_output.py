"""Publish domain-valued rollout decisions for sequence preprocessing."""

import json
from contextlib import contextmanager, nullcontext
from itertools import islice
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import polars as pl

from sequifier.contract_versions import ContractVersions
from sequifier.fingerprints import file_checksum, fingerprint
from sequifier.io.action_record import action_record_layout
from sequifier.io.actions import decode_action
from sequifier.io.atomic import atomic_path, write_json_atomic
from sequifier.io.sequence import CONTEXT_COLUMN, CONTEXT_RECORDS, ContextObservation
from sequifier.io.values import portable_value
from sequifier.trajectory.tables import ACTION_RECORD, PREVIOUS_ACTION_RECORD


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
):
    row = {
        "sequenceId": sequence_coordinate(identity),
        "episodeId": identity,
        "itemPosition": step,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "bootstrap_allowed": not terminated,
        "__group": group,
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


def trajectory_rows(trajectory):
    from sequifier.config.policy_config import ResolvedPolicyBinding
    from sequifier.trajectory.importers import sequence_from_trajectory

    execution = trajectory.metadata.get("policy_execution")
    episode = None
    if execution and execution.get("config"):
        policy = ResolvedPolicyBinding.model_validate(execution["config"])
        episode = sequence_from_trajectory(trajectory, policy)
    origin = trajectory.metadata.get("seed_origin", {})
    group = (
        fingerprint(
            {key: origin[key] for key in ("dataset", "sequenceId", "itemPosition")}
        )
        if {"dataset", "sequenceId", "itemPosition"} <= origin.keys()
        else None
    )
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
                    observation={
                        name: values[index]
                        for name, values in trajectory.seed.observations.items()
                    }
                )
                for index, valid in enumerate(trajectory.seed.attention_valid_mask[:-1])
                if valid
            ]
            if context:
                row[CONTEXT_COLUMN] = CONTEXT_RECORDS.dump_json(tuple(context)).decode()
        yield row


def write_part(manifest, rows, *, inference=False):
    """Immutable shards let a checkpoint retain an exact prefix across retries."""
    from sequifier.io.inference_workspace import inference_output

    path = Path(manifest)
    parts = path.with_name(path.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    table = pl.from_dicts(list(rows), infer_schema_length=None)
    with TemporaryDirectory(dir=parts) as temp:
        shard = Path(temp) / "part.parquet"
        table.write_parquet(shard)
        checksum = file_checksum(shard)
        destination = parts / (checksum + ".parquet")
        with inference_output(destination) if inference else nullcontext(
            destination
        ) as staged:
            with atomic_path(staged, durable=True) as temporary:
                copyfile(shard, temporary)
    return {
        "path": str(destination.relative_to(path.parent)),
        "sha256": checksum,
        "decisions": len(table),
        "sequences": table["sequenceId"].unique().sort().to_list(),
    }


def write_parts(manifest, rows, *, inference=False):
    rows = iter(rows)
    while batch := list(islice(rows, 64)):
        yield write_part(manifest, batch, inference=inference)


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
        shards = list(
            write_parts(manifest, trajectory_rows(trajectory), inference=True)
        )
        with inference_output(manifest) as staged:
            write_json_atomic(
                staged,
                manifest_values(
                    shards,
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
                ),
            )


@contextmanager
def supervised_manifest_input(path, sequence_view="decisions", *, dependencies=None):
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
            scan = pl.from_dicts(
                list(trajectory_rows(trajectory)), infer_schema_length=None
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
            observations = manifest.get("observation_columns") or manifest.get(
                "policy_schema", {}
            ).get("observation_columns")
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
            coordinates = ["sequenceId", "itemPosition", "episodeId", "__group"]
            static = manifest.get(
                "static_columns",
                manifest.get("policy_schema", {}).get("static_columns", []),
            )
            columns = coordinates + observations + list(static)
            final = final.with_columns(
                (pl.col("itemPosition") + 1).alias("itemPosition"),
                *[pl.col("__next_" + c).alias(c) for c in observations],
            )
            scan = pl.concat(
                [scan.select(columns), final.select(columns)], how="vertical_relaxed"
            )
        scan.sink_parquet(table)
        if resolved is not None:
            resolved.dependencies.revalidate()
        yield table, manifest
