"""Adapters from tables, prepared histories and rollout logs to sequence data."""

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import polars as pl

from sequifier.config.metadata import load_dataset_metadata
from sequifier.config.policy_config import PolicySchemaConfig
from sequifier.config.policy_train_config import (
    PolicyTableColumns,
    SequenceSourceConfig,
)
from sequifier.io.preprocessing import denormalize_values
from sequifier.io.sequence import (
    SequenceEpisode,
    TransitionColumns,
    ordered_sequence_rows,
)
from sequifier.io.values import domain_scalar, numeric_scalar
from sequifier.trajectory.schema import Trajectory, validate_trajectory
from sequifier.trajectory.store import load_trajectory


def _checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_paths(source: SequenceSourceConfig) -> list[Path]:
    if source.manifest is None:
        paths = [Path(path) for path in source.paths]
    else:
        manifest_path = Path(source.manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("version") != 1:
            raise ValueError("source manifest must have version 1")
        sequence_manifest = manifest.get("format") == "sequifier.sequence-data"
        entries = manifest.get("shards" if sequence_manifest else "episodes")
        if not isinstance(entries, list) or not entries:
            raise ValueError("source manifest requires a nonempty episodes list")
        paths = []
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("path"), str)
                or not entry["path"]
            ):
                raise ValueError("each manifest entry requires a nonempty path")
            path = manifest_path.parent / entry["path"]
            if sequence_manifest and _checksum(path) != entry.get("sha256"):
                raise ValueError(f"sequence shard checksum disagrees with {path}")
            if (
                "episode_id" in entry
                and load_trajectory(path).episode_id != entry["episode_id"]
            ):
                raise ValueError(f"manifest episode ID disagrees with {path}")
            paths.append(path)
    paths = [path.resolve() for path in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("sources require nonempty, unique paths")
    return paths


def sequence_from_trajectory(
    trajectory: Trajectory, policy: PolicySchemaConfig, action_semantics="policy"
) -> SequenceEpisode:
    episode = validate_trajectory(trajectory)
    if (
        episode.format != "sequifier.scalar-policy-trajectory"
        and action_semantics != "executed_predictions"
    ):
        raise ValueError(
            "forecast sources require explicit action_semantics='executed_predictions'"
        )
    if episode.prediction_columns != [policy.action_column]:
        raise ValueError("recorded actions do not match the policy action column")
    steps = episode.transitions
    return SequenceEpisode(
        episode_id=episode.episode_id,
        item_positions=tuple(step.step_id for step in steps),
        observations={
            column: tuple(step.observation[column] for step in steps)
            for column in episode.seed.observations
        },
        actions=tuple(step.predictions[policy.action_column] for step in steps),
        static_inputs=episode.seed.static_inputs,
        transitions=TransitionColumns(
            **{
                field: tuple(getattr(step, attribute) for step in steps)
                for field, attribute in {
                    "rewards": "reward",
                    "terminated": "terminated",
                    "truncated": "truncated",
                    "successors": "successor_observation",
                    "bootstrap_allowed": "bootstrap_allowed",
                }.items()
            }
        ),
    )


@dataclass(frozen=True)
class TrajectoryReference:
    path: Path
    policy: PolicySchemaConfig
    action_semantics: str = "policy"
    namespace: str = ""

    def load(self) -> SequenceEpisode:
        episode = sequence_from_trajectory(
            load_trajectory(self.path), self.policy, self.action_semantics
        )
        return episode.model_copy(
            update={"episode_id": self.namespace + episode.episode_id}
        )


def _scan(paths: tuple[Path, ...]):
    scans = []
    for path in paths:
        if path.suffix.lower() == ".csv":
            scans.append(pl.scan_csv(path, infer_schema_length=None))
        elif path.suffix.lower() == ".parquet":
            scans.append(pl.scan_parquet(path))
        else:
            raise ValueError("table sources require CSV or Parquet files")
    return pl.concat(scans, how="vertical_relaxed")


def _prepared_table(
    paths: tuple[Path, ...], source: SequenceSourceConfig
) -> pl.DataFrame:
    if source.metadata_config_path is None:
        raise ValueError("prepared histories require metadata_config_path")
    signatures = tuple(
        (path.stat().st_mtime_ns, path.stat().st_size, path.stat().st_ino)
        for path in (*paths, Path(source.metadata_config_path))
    )
    return _prepared_table_contents(paths, source.metadata_config_path, signatures)


@lru_cache(maxsize=1)
def _prepared_table_contents(
    paths: tuple[Path, ...], metadata_path: str, signatures: tuple
) -> pl.DataFrame:
    """Recover unique positions from existing windows, validating overlaps.

    Split target bounds select decisions. A preceding action may be recovered
    from stored context, but observations outside the split are not targets.
    """
    from sequifier.helpers import PANDAS_TO_TORCH_TYPES, read_data
    from sequifier.io.pt_payload import load_pt_payload, tabular_tensor_batch

    metadata = load_dataset_metadata(metadata_path)
    if metadata.depth_layouts:
        raise ValueError(
            "policy table import currently supports scalar prepared columns"
        )
    rows = {}
    for path in paths:
        if path.suffix.lower() == ".pt":
            batch = load_pt_payload(path, n_classes=metadata.n_classes)
        else:
            data = read_data(str(path), path.suffix.lower().lstrip("."))
            batch = tabular_tensor_batch(
                data,
                {
                    column: PANDAS_TO_TORCH_TYPES[dtype]
                    for column, dtype in metadata.column_data_types.items()
                },
                sorted(set(data["inputCol"].to_list())),
                metadata.window_length,
            )
        sequences, ids, starts, pads = (
            batch.sequences,
            batch.sequence_ids,
            batch.start_item_positions,
            batch.left_pad_lengths,
        )
        lows, highs = batch.split_start_item_positions, batch.split_end_item_positions
        for row in range(len(ids)):
            for offset in range(
                int(pads[row]), next(iter(sequences.values())).shape[1]
            ):
                position = int(starts[row]) + offset
                key = (int(ids[row]), position)
                values = {
                    column: tensor[row, offset].item()
                    for column, tensor in sequences.items()
                }
                record = {
                    "sequenceId": key[0],
                    "itemPosition": position,
                    **values,
                    "__decision": int(lows[row]) <= position < int(highs[row]),
                }
                if key in rows and rows[key] != record:
                    raise ValueError(
                        "prepared history windows disagree on overlapping positions or split bounds"
                    )
                rows[key] = record
    if not rows:
        raise ValueError("prepared source contains no valid positions")
    return pl.DataFrame(list(rows.values())).sort(["sequenceId", "itemPosition"])


@dataclass(frozen=True)
class TableEpisodeReference:
    paths: tuple[Path, ...]
    source: SequenceSourceConfig
    policy: PolicySchemaConfig
    sequence_id: str | int
    prepared: bool = False

    def load(self) -> SequenceEpisode:
        columns = self.source.columns
        data = (
            _prepared_table(self.paths, self.source).filter(
                pl.col(columns.sequence_id) == self.sequence_id
            )
            if self.prepared
            else _scan(self.paths)
            .filter(pl.col(columns.sequence_id) == self.sequence_id)
            .collect()
        )
        data = ordered_sequence_rows(data, columns.item_position)
        metadata = (
            load_dataset_metadata(self.source.metadata_config_path)
            if self.source.metadata_config_path
            else None
        )
        model_encoded = self.prepared or self.source.encoding == "model"

        def decode(value, column):
            value = domain_scalar(value, column)
            if model_encoded and metadata is not None:
                if column in metadata.id_maps:
                    inverse = {
                        encoded: label
                        for label, encoded in metadata.id_maps[column].items()
                    }
                    if (
                        isinstance(value, str)
                        or value != int(value)
                        or int(value) not in inverse
                    ):
                        raise ValueError(f"{column}: encoded ID has no domain label")
                    value = inverse[int(value)]
                    if column == (columns.action or self.policy.action_column):
                        if isinstance(value, str) and not isinstance(
                            self.policy.action_values[0], str
                        ):
                            try:
                                value = numeric_scalar(float(value), column)
                            except ValueError:
                                pass
                        matches = [
                            action
                            for action in self.policy.action_values
                            if action == value or str(action) == str(value)
                        ]
                        if len(matches) == 1:
                            value = matches[0]
                elif (
                    metadata.normalize_real_columns
                    and column in metadata.selected_columns_statistics
                ):
                    value = denormalize_values(
                        numeric_scalar(value, column),
                        metadata.selected_columns_statistics[column],
                    )
            return value

        preceding = None
        if self.prepared:
            decisions = data.filter(pl.col("__decision"))
            if decisions.is_empty():
                raise ValueError("prepared episode contains no split-owned decisions")
            first = decisions[columns.item_position][0]
            previous = data.filter(pl.col(columns.item_position) == first - 1)
            if len(previous):
                preceding = decode(
                    previous[columns.action or self.policy.action_column][0],
                    columns.action or self.policy.action_column,
                )
            data = decisions
        static_data = None
        if self.source.static_data_path is not None:
            static_data = (
                _scan((Path(self.source.static_data_path),))
                .filter(pl.col(columns.sequence_id) == self.sequence_id)
                .collect()
            )
        return table_episode(
            data,
            self.policy,
            columns,
            sequence_id=self.sequence_id,
            preceding=preceding,
            static_data=static_data,
            episode_end=self.source.episode_end,
            episode_namespace=self.source.episode_namespace,
            decode=decode,
        )


def table_episode(
    data,
    policy,
    columns,
    *,
    sequence_id,
    preceding=None,
    static_data=None,
    episode_end="unknown",
    episode_namespace="",
    decode=domain_scalar,
) -> SequenceEpisode:
    """Canonical table-to-episode validation shared by training and inference."""
    data = ordered_sequence_rows(data, columns.item_position)
    rows = list(data.iter_rows(named=True))
    if not rows:
        raise ValueError("sequence source changed while loading")
    action_column = columns.action or policy.action_column
    observations, static_inputs = (
        {name: mapping.get(name, name) for name in names}
        for names, mapping in (
            (policy.observation_columns, columns.observations),
            (policy.static_columns, columns.static_inputs),
        )
    )
    inputs = set(observations.values())
    if static_data is None:
        inputs.update(static_inputs.values())
    if action_column in inputs or columns.previous_action == action_column:
        raise ValueError(
            "current action labels require distinct input/previous-action columns"
        )
    if (
        set(columns.observations) - observations.keys()
        or set(columns.static_inputs) - static_inputs.keys()
    ):
        raise ValueError("table mappings must name declared policy features")
    required = (
        inputs
        | {columns.item_position, action_column}
        | set(columns.successors.values())
    )
    required.update(
        filter(
            None,
            (
                columns.previous_action,
                columns.bootstrap_allowed,
                columns.group,
                columns.episode_id,
            ),
        )
    )
    if required - set(data.columns):
        raise ValueError(
            f"policy table is missing columns: {sorted(required - set(data.columns))}"
        )

    def values(column, decoder=decode):
        return tuple(decoder(row[column], column) for row in rows)

    def optional_values(column, decoder=decode):
        if (
            column is None
            or column not in data.columns
            or data[column].null_count() == len(rows)
        ):
            return None
        return values(column, decoder)

    def constant(entries, label):
        unique = set(entries)
        if len(unique) != 1:
            raise ValueError(f"{label} must remain constant within an episode")
        return next(iter(unique))

    positions = values(columns.item_position, lambda value, _: value)
    actions = values(action_column)
    if columns.previous_action is not None:
        preceding_actions = values(
            columns.previous_action,
            lambda value, _: None if value is None else decode(value, action_column),
        )
        if preceding_actions[1:] != actions[:-1]:
            raise ValueError(
                "previous-action column must lag executed actions by exactly one decision"
            )
        preceding = preceding_actions[0]
    if static_data is not None:
        if len(static_data) != 1:
            raise ValueError("static data requires exactly one row per sequenceId")
        static_rows = list(static_data.iter_rows(named=True))
    else:
        static_rows = rows
    static = {
        name: constant(
            (decode(row[column], column) for row in static_rows),
            f"static input {name}",
        )
        for name, column in static_inputs.items()
    }
    # Strict boolean and length checks belong to TransitionColumns/SequenceEpisode.
    transitions = {
        field: optional_values(getattr(columns, field), lambda value, _: value)
        for field in ("terminated", "truncated", "bootstrap_allowed")
    }
    transitions = TransitionColumns(**transitions).model_dump(exclude_none=True)
    declared = episode_end
    if declared != "unknown":
        current = transitions.get(declared)
        if current is not None and not current[-1]:
            raise ValueError("episode_end conflicts with the recorded final boundary")
        transitions[declared] = (False,) * (len(rows) - 1) + (True,)
    successors = None
    if columns.successors:
        if columns.successors.keys() != observations.keys():
            raise ValueError("successor mapping must cover every observation column")
        successors = []
        for row in rows:
            present = [
                row[column] is not None for column in columns.successors.values()
            ]
            if any(present) and not all(present):
                raise ValueError("a missing successor must mask all observation fields")
            successors.append(
                {
                    name: decode(row[column], observations[name])
                    for name, column in columns.successors.items()
                }
                if all(present)
                else None
            )
    groups = optional_values(columns.group, domain_scalar)
    identity = (
        constant(values(columns.episode_id, domain_scalar), "episode ID")
        if columns.episode_id
        else sequence_id
    )
    return SequenceEpisode(
        episode_id=episode_namespace + str(identity),
        item_positions=positions,
        observations={name: values(column) for name, column in observations.items()},
        actions=actions,
        static_inputs=static,
        starts_episode=positions[0] == 0 and preceding is None,
        previous_action=preceding,
        group_id=str(constant(groups, "collection group"))
        if groups is not None
        else None,
        transitions=TransitionColumns(
            **transitions,
            rewards=optional_values(
                columns.reward,
                lambda value, column: numeric_scalar(decode(value, column), "reward"),
            ),
            successors=tuple(successors) if successors is not None else None,
        ),
    )


def source_episodes(source: SequenceSourceConfig, policy: PolicySchemaConfig):
    if source.split_index is not None:
        from sequifier.config.policy_train_config import resolve_sequence_source

        source = resolve_sequence_source(source, Path.cwd())
    paths = tuple(source_paths(source))
    if source.manifest is not None:
        manifest = json.loads(Path(source.manifest).read_text(encoding="utf-8"))
        if manifest.get("format") == "sequifier.sequence-data":
            recorded_columns = PolicyTableColumns.model_validate(manifest["columns"])
            if (
                source.columns != PolicyTableColumns()
                and source.columns != recorded_columns
            ):
                raise ValueError(
                    "source mappings disagree with the committed sequence manifest"
                )
            if source.encoding != "raw" or source.episode_end != "unknown":
                raise ValueError(
                    "committed sequence data already declares its encoding and boundaries"
                )
            source = source.model_copy(
                update={"columns": recorded_columns, "format": "table"}
            )
    kind = source.format
    if kind == "auto":
        suffixes = {path.suffix.lower() for path in paths}
        if suffixes <= {".json"}:
            kind = "trajectory"
        elif suffixes <= {".csv", ".parquet"}:
            kind = "table"
        elif suffixes == {".pt"}:
            kind = "prepared"
        else:
            raise ValueError(
                "select a source format explicitly for mixed or unsupported files"
            )
    if kind == "trajectory":
        return [
            TrajectoryReference(
                path, policy, source.action_semantics, source.episode_namespace
            )
            for path in paths
        ]
    prepared = kind == "prepared"
    data = _prepared_table(paths, source).lazy() if prepared else _scan(paths)
    columns = source.columns
    required = {columns.sequence_id, columns.item_position}
    if required - set(data.collect_schema().names()):
        raise ValueError(
            "policy tables require sequenceId and itemPosition, or explicit column mappings"
        )
    ids = (
        data.select(columns.sequence_id)
        .unique(maintain_order=True)
        .collect()[columns.sequence_id]
        .to_list()
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, str)) for value in ids
    ):
        raise ValueError("sequence IDs must be non-null integers or strings")
    if source.static_data_path:
        static_ids = (
            _scan((Path(source.static_data_path),))
            .select(columns.sequence_id)
            .collect()[columns.sequence_id]
            .to_list()
        )
        if len(static_ids) != len(set(static_ids)) or not set(ids) <= set(static_ids):
            raise ValueError(
                "static data must cover every selected sequence ID without duplicates"
            )
    return [
        TableEpisodeReference(paths, source, policy, identity, prepared)
        for identity in ids
    ]


def import_sequences(config) -> Path:
    """Commit canonical columnar episodes as checksummed raw Parquet shards."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = config.resolved_source()
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
        successors={
            name: "__next_" + name for name in config.policy.observation_columns
        },
    )
    fields = config.policy.observation_columns + config.policy.static_columns
    reserved = {
        value for value in columns.model_dump().values() if isinstance(value, str)
    } | set(columns.successors.values())
    if set(fields) & reserved:
        raise ValueError(
            "policy feature names collide with sequence import metadata columns"
        )
    references = source_episodes(source, config.policy)
    if not references:
        raise ValueError("sequence import requires at least one episode")
    real = (
        set(fields)
        - set(config.policy.categorical_columns)
        - config.policy.categorical_n_classes.keys()
    )
    action_type = (
        pa.string() if isinstance(config.policy.action_values[0], str) else pa.float64()
    )
    types = {name: pa.float64() for name in real | {"__reward"}}
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
                    "sha256": _checksum(stage / name),
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
            "policy_schema": config.policy.model_dump(mode="json"),
            "shards": shards,
            "provenance": {
                "source": source.model_dump(mode="json"),
                "files": [
                    {"path": str(path), "sha256": _checksum(path)}
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
