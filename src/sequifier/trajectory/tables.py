"""Indexed table episode readers and canonical table-to-episode validation."""

from __future__ import annotations

import tempfile
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from sequifier.config.metadata import DatasetMetadata
from sequifier.config.policy_inputs import PolicyEpisodeSpec
from sequifier.config.policy_train_config import SequenceSourceConfig
from sequifier.io.images import ImageStore
from sequifier.io.preprocessing import denormalize_values
from sequifier.io.sequence import (
    CONTEXT_COLUMN,
    CONTEXT_RECORDS,
    ContextObservation,
    PreparedEpisode,
    SequenceEpisode,
    TransitionColumns,
    ordered_sequence_rows,
)
from sequifier.io.values import domain_scalar, numeric_scalar
from sequifier.trajectory.prepared import prepared_table

if TYPE_CHECKING:
    from sequifier.io.indexed_sequences import IndexedImageStore
    from sequifier.trajectory.sources import ResolvedSequenceSource

ACTION_RECORD = "__executed_action"
PREVIOUS_ACTION_RECORD = "__previous_action"
PREVIOUS_ACTION_PREFIX = "__previous_action_"
SUCCESSOR_PREFIX = "__next_"
BOOTSTRAP_COLUMN = "bootstrap_allowed"
GROUP_COLUMN = "__group"
EPISODE_ID_COLUMN = "episodeId"
REWARD_COLUMN = "reward"
TRANSITION_FLAGS = ("terminated", "truncated", BOOTSTRAP_COLUMN)


def scan_tables(paths: tuple[Path, ...]):
    scans = []
    for path in paths:
        if path.suffix.lower() == ".csv":
            scan = pl.scan_csv(path, infer_schema_length=None)
        elif path.suffix.lower() == ".parquet":
            scan = pl.scan_parquet(path)
        else:
            raise ValueError("table sources require CSV or Parquet files")
        schema = scan.collect_schema()
        if "sequenceId" in schema:
            if not schema["sequenceId"].is_integer():
                raise ValueError("sequenceId must contain non-null integers")
            scan = scan.with_columns(pl.col("sequenceId").cast(pl.Int64))
        scans.append(scan)
    if any(CONTEXT_COLUMN in scan.collect_schema() for scan in scans):
        scans = [
            scan
            if CONTEXT_COLUMN in scan.collect_schema()
            else scan.with_columns(pl.lit(None, dtype=pl.String).alias(CONTEXT_COLUMN))
            for scan in scans
        ]
        columns = scans[0].collect_schema().names()
        scans = [scan.select(columns) for scan in scans]
    return pl.concat(scans, how="vertical_relaxed")


class IndexedTable:
    """Canonical sorted storage with direct row-group access per episode.

    The temporary directory lives as long as its episode references. CSV parsing
    and sorting happen once; cache misses never filter the original source.
    """

    def __init__(self, data, sequence_column):
        import pyarrow.parquet as pq

        self.directory = tempfile.TemporaryDirectory(prefix="sequifier-episodes-")
        self.path = Path(self.directory.name) / "episodes.parquet"
        data.sort(sequence_column, maintain_order=True).sink_parquet(
            self.path, row_group_size=8192
        )
        table = pq.ParquetFile(self.path)
        self.offsets = [0]
        for i in range(table.num_row_groups):
            self.offsets.append(self.offsets[-1] + table.metadata.row_group(i).num_rows)
        self.ranges = {}
        offset = 0
        for batch in table.iter_batches(columns=[sequence_column]):
            for identity in batch.column(0).to_pylist():
                if isinstance(identity, bool) or not isinstance(identity, int):
                    raise ValueError("sequenceId must contain non-null integers")
                if identity not in self.ranges:
                    self.ranges[identity] = [offset, offset + 1]
                else:
                    self.ranges[identity][1] = offset + 1
                offset += 1
        table.close()

    def load(self, identity):
        import pyarrow.parquet as pq

        start, end = self.ranges[identity]
        first = bisect_right(self.offsets, start) - 1
        last = bisect_right(self.offsets, end - 1) - 1
        with pq.ParquetFile(self.path) as table:
            rows = table.read_row_groups(list(range(first, last + 1)))
        return pl.from_arrow(rows.slice(start - self.offsets[first], end - start))


class IndexedEpisodeTable:
    """Read split-owned episodes directly from the committed sequence store."""

    def __init__(self, view, context_length):
        self.context_length = context_length
        self.view = view
        self.ranges = view.split_bounds()
        if not self.ranges:
            raise ValueError(
                f"{view.path}: policy episodes require recorded split membership"
            )

    def load(self, identity):
        start, end = self.ranges[identity]
        rows = self.view.read(identity, start - self.context_length, end)
        record = self.view.dataset["sequences"][str(identity)]
        for field, column in (
            ("episode_id", EPISODE_ID_COLUMN),
            ("group_id", GROUP_COLUMN),
        ):
            if record.get(field) is not None and column not in rows.columns:
                rows = rows.with_columns(pl.lit(record[field]).alias(column))
        return rows.with_columns((pl.col("itemPosition") >= start).alias("__decision"))


def model_decoder(metadata, actions):
    """Decode model-encoded scalars using one fixed set of inverse codecs."""
    inverse_maps = (
        {
            column: {encoded: label for label, encoded in mapping.items()}
            for column, mapping in metadata.id_maps.items()
        }
        if metadata is not None
        else {}
    )

    def decode(value, column):
        if (
            column.startswith(("__image:", "__image_mask:"))
            or column == "__preconditioning_image"
        ):
            return value
        if isinstance(value, (list, tuple)):
            return [decode(item, column) for item in value]
        if isinstance(value, bool) or isinstance(value, dict):
            return value
        value = domain_scalar(value, column)
        if metadata is not None:
            if column in inverse_maps:
                inverse = inverse_maps[column]
                if (
                    isinstance(value, str)
                    or value != int(value)
                    or int(value) not in inverse
                ):
                    from sequifier.special_tokens import SPECIAL_TOKEN_IDS

                    if (
                        isinstance(value, int)
                        and value in SPECIAL_TOKEN_IDS.labels_by_id
                    ):
                        return SPECIAL_TOKEN_IDS.labels_by_id[value]
                    raise ValueError(f"{column}: encoded ID has no domain label")
                value = inverse[int(value)]
                if column in actions and actions[column].type == "categorical":
                    action_values = actions[column].values
                    if isinstance(value, str) and not isinstance(action_values[0], str):
                        try:
                            value = numeric_scalar(float(value), column)
                        except ValueError:
                            pass
                    matches = [
                        action
                        for action in action_values
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

    return decode


@dataclass(frozen=True)
class TableEpisodeReference:
    source: SequenceSourceConfig
    policy: PolicyEpisodeSpec
    sequence_id: int
    table: IndexedTable | IndexedEpisodeTable
    metadata: DatasetMetadata | None = None
    prepared: bool = False
    static_table: IndexedTable | None = None
    image_store: ImageStore | IndexedImageStore | None = None

    @property
    def content_key(self):
        if isinstance(self.table, IndexedEpisodeTable):
            return self.table.view.dataset["sequences"][str(self.sequence_id)].get(
                "content_key"
            )
        return None

    @property
    def used_for_fitting(self) -> bool:
        return (
            isinstance(self.table, IndexedEpisodeTable)
            and self.sequence_id in self.table.view.fitted_sequences
        )

    def load(self) -> SequenceEpisode:
        data = ordered_sequence_rows(self.table.load(self.sequence_id))
        decode = model_decoder(
            self.metadata if self.prepared or self.source.encoding == "model" else None,
            self.policy.actions,
        )

        owned_start = None
        if self.prepared:
            decisions = data.filter(pl.col("__decision"))
            if decisions.is_empty():
                raise ValueError("prepared episode contains no split-owned decisions")
            owned_start = decisions["itemPosition"][0]
            data = data.filter(
                pl.col("itemPosition") >= owned_start - self.policy.context_length
            )
        static_data = None
        if self.static_table is not None:
            static_data = self.static_table.load(self.sequence_id)
        if self.image_store is not None:
            import torch

            images, masks = self.image_store.gather(
                torch.tensor([int(self.sequence_id)]),
                torch.tensor([data["itemPosition"].to_list()]),
                torch.ones((1, len(data)), dtype=torch.bool),
            )
            for name, value in {**images, **masks}.items():
                if name == "__preconditioning_image":
                    if static_data is None:
                        data = data.with_columns(
                            pl.Series(name, [value[0].tolist()] * len(data))
                        )
                    else:
                        static_data = static_data.with_columns(
                            pl.Series(name, [value[0].tolist()])
                        )
                else:
                    data = data.with_columns(pl.Series(name, value[0].tolist()))
        return table_episode(
            data,
            self.policy,
            sequence_id=self.sequence_id,
            static_data=static_data,
            owned_start=owned_start,
            episode_end=self.source.episode_end,
            episode_namespace=self.source.episode_namespace,
            decode=decode,
            raw_columns=self.metadata.indexed_passthrough
            if self.metadata is not None
            else (),
        )


def table_episode(
    data,
    policy,
    *,
    sequence_id,
    preceding_row=None,
    owned_start=None,
    static_data=None,
    episode_end="unknown",
    episode_namespace="",
    decode=domain_scalar,
    raw_columns=(),
) -> SequenceEpisode:
    """Read an episode using the policy's canonical feature names."""
    data = ordered_sequence_rows(data)
    if decode is domain_scalar:

        def identity_decode(value, column):
            return value

        decode = identity_decode

    if data.is_empty():
        raise ValueError("sequence source changed while loading")

    observations = policy.observation_columns
    static_inputs = policy.static_columns
    action_names = policy.action_columns
    structured_actions = ACTION_RECORD in data.columns
    structured_previous = PREVIOUS_ACTION_RECORD in data.columns
    action_columns = {ACTION_RECORD} if structured_actions else set(action_names)
    previous_columns = [PREVIOUS_ACTION_PREFIX + name for name in action_names]
    scalar_previous = not structured_previous and any(
        column in data.columns for column in previous_columns
    )
    if structured_previous:
        previous_inputs = {PREVIOUS_ACTION_RECORD}
    elif scalar_previous:
        previous_inputs = set(previous_columns)
    else:
        previous_inputs = set()
    successor_columns = [SUCCESSOR_PREFIX + name for name in observations]
    has_successors = bool(set(successor_columns) & set(data.columns))
    raw_successors = set(successor_columns) & set(raw_columns)
    if raw_successors and policy.input_encoding == "model":
        raise ValueError(
            f"Model-encoded policy input requires encoded successors: {sorted(raw_successors)}"
        )
    inputs = set(observations)
    if static_data is None:
        inputs.update(static_inputs)
    if action_columns & (inputs | previous_inputs):
        raise ValueError(
            "current action labels require distinct input/previous-action columns"
        )
    if previous_inputs & inputs:
        raise ValueError("previous actions require distinct input columns")
    if has_successors and set(successor_columns) & (
        inputs | action_columns | previous_inputs
    ):
        raise ValueError("successors require distinct input and action columns")
    required = inputs | {"itemPosition"} | action_columns | previous_inputs
    if has_successors:
        required.update(successor_columns)
    if missing := required - set(data.columns):
        raise ValueError(f"policy table is missing columns: {sorted(missing)}")
    for record in (ACTION_RECORD, PREVIOUS_ACTION_RECORD):
        if (
            record in data.columns
            and data.schema[record] != pl.Null
            and not isinstance(data.schema[record], pl.Struct)
        ):
            raise ValueError(f"{record}: action records require a structured column")

    def canonical_action(value):
        if value is None:
            return None
        return {
            name: entry
            for name, entry in value.items()
            if name in policy.actions or entry is not None
        }

    def action_at(row):
        if structured_actions:
            record = canonical_action(row[ACTION_RECORD])
            if record is None:
                raise ValueError("A decision requires a recorded action")
            return record
        return {name: decode(row[name], name) for name in action_names}

    def values(column, decoder=decode):
        return tuple(decoder(value, column) for value in data[column].to_list())

    def optional_values(column, decoder=decode):
        if column not in data.columns or data[column].null_count() == len(data):
            return None
        return values(column, decoder)

    def constant(entries, label):
        entries = list(entries)
        if not entries or any(value != entries[0] for value in entries[1:]):
            raise ValueError(f"{label} must remain constant within an episode")
        return entries[0]

    positions = tuple(data["itemPosition"])
    actions = tuple(action_at(row) for row in data.iter_rows(named=True))
    preceding = action_at(preceding_row) if preceding_row is not None else None
    if scalar_previous:
        previous = tuple(
            dict(zip(action_names, entries, strict=True))
            for entries in data.select(previous_columns).iter_rows()
        )
        if previous[1:] != actions[:-1]:
            raise ValueError(
                "previous actions must lag recorded actions by one decision"
            )
        preceding = (
            None
            if all(value is None for value in previous[0].values())
            else previous[0]
        )
    if structured_previous:
        previous = tuple(map(canonical_action, data[PREVIOUS_ACTION_RECORD]))
        if previous[1:] != actions[:-1]:
            raise ValueError(
                "previous-action column must lag executed actions by exactly one decision"
            )
        preceding = previous[0]
    if static_data is not None:
        if len(static_data) != 1:
            raise ValueError("static data requires exactly one row per sequenceId")
        if missing := set(static_inputs) - set(static_data.columns):
            raise ValueError(f"static table is missing columns: {sorted(missing)}")
    else:
        static_data = data

    def feature_value(value, name):
        return value if policy.input_encoding == "model" else decode(value, name)

    static = {
        name: constant(
            (feature_value(value, name) for value in static_data[name].to_list()),
            f"static input {name}",
        )
        for name in static_inputs
    }
    groups = optional_values(GROUP_COLUMN, domain_scalar)
    identity = (
        constant(values(EPISODE_ID_COLUMN, domain_scalar), "episode ID")
        if EPISODE_ID_COLUMN in data.columns
        and data[EPISODE_ID_COLUMN].null_count() < len(data)
        else sequence_id
    )

    def reward_value(value, column):
        return numeric_scalar(
            value if column in raw_columns else decode(value, column), "reward"
        )

    context = ()
    if CONTEXT_COLUMN in data.columns and (stored := data[CONTEXT_COLUMN][0]):
        context = CONTEXT_RECORDS.validate_json(stored)
        context = tuple(
            step.model_copy(
                update={
                    "observation": {
                        name: feature_value(value, name)
                        for name, value in step.observation.items()
                    }
                }
            )
            for step in context
        )
    if owned_start is not None and owned_start > positions[0]:
        offset = owned_start - positions[0]
        prefix = data.head(offset)
        if any(
            name in prefix.columns and prefix[name].any()
            for name in ("terminated", "truncated")
        ):
            raise ValueError("Prepared history crosses an episode boundary")
        first = max(0, offset - policy.context_length + 1)
        if positions[0] and preceding is None and policy.previous_action_inputs:
            first = max(1, first)
        frames = []
        for i, row in enumerate(prefix.slice(first).iter_rows(named=True), first):
            frames.append(
                ContextObservation(
                    observation={
                        name: feature_value(row[name], name) for name in observations
                    },
                    previous_action=actions[i - 1] if i else preceding,
                    reward=reward_value(row[REWARD_COLUMN], REWARD_COLUMN)
                    if row.get(REWARD_COLUMN) is not None
                    else None,
                )
            )
        context = (
            (*context, *frames)[-(policy.context_length - 1) :]
            if policy.context_length > 1
            else ()
        )
        preceding = actions[offset - 1]
        actions = actions[offset:]
        positions = positions[offset:]
        data = data.slice(offset)

    # Strict boolean and length checks belong to TransitionColumns/SequenceEpisode.
    transitions = {
        field: optional_values(field, lambda value, _: value)
        for field in TRANSITION_FLAGS
    }
    transitions = TransitionColumns(**transitions).model_dump(exclude_none=True)
    declared = episode_end
    if declared != "unknown":
        current = transitions.get(declared)
        if current is not None and not current[-1]:
            raise ValueError("episode_end conflicts with the recorded final boundary")
        transitions[declared] = (False,) * (len(data) - 1) + (True,)
    successors = None
    if has_successors:
        successors = []
        for entries in data.select(successor_columns).iter_rows():
            present = [value is not None for value in entries]
            if any(present) and not all(present):
                raise ValueError("a missing successor must mask all observation fields")
            successors.append(
                {
                    name: (
                        value
                        if SUCCESSOR_PREFIX + name in raw_successors
                        else feature_value(value, name)
                    )
                    for name, value in zip(observations, entries, strict=True)
                }
                if all(present)
                else None
            )
    episode = SequenceEpisode(
        sequence_id=int(sequence_id),
        episode_id=episode_namespace + str(identity),
        item_positions=positions,
        observations={name: values(name, feature_value) for name in observations},
        actions=actions,
        static_inputs=static,
        starts_episode=positions[0] == 0 and preceding is None,
        previous_action=preceding,
        group_id=str(constant(groups, "collection group"))
        if groups is not None
        else None,
        transitions=TransitionColumns(
            **transitions,
            rewards=optional_values(REWARD_COLUMN, reward_value),
            successors=tuple(successors) if successors is not None else None,
        ),
    )

    return (
        PreparedEpisode(**episode.model_dump(), context=context) if context else episode
    )


def table_episodes(resolved: ResolvedSequenceSource, policy: PolicyEpisodeSpec):
    from sequifier.io.indexed_sequences import IndexedView, is_indexed_view

    paths = resolved.paths
    source = resolved.declaration
    prepared = resolved.kind == "prepared"
    view = (
        IndexedView.load(paths[0])
        if prepared and len(paths) == 1 and is_indexed_view(paths[0])
        else None
    )
    if set(source.columns.observations) - set(policy.observation_columns) or set(
        source.columns.static_inputs
    ) - set(policy.static_columns):
        raise ValueError("table columns must name declared policy features")
    static_path, image_store = resolved.static_path, None
    if view is not None:
        table, metadata = (
            IndexedEpisodeTable(view, policy.context_length),
            view.metadata(),
        )
        for label, configured, stored in (
            ("metadata", resolved.metadata_path, view.metadata_path),
            ("static data", static_path, view.static_path),
        ):
            if configured and (
                not stored or Path(configured).resolve() != Path(stored).resolve()
            ):
                raise ValueError(f"Indexed sources resolve {label} from the view")
        static_path = view.static_path
        if not metadata.indexed_inline_media:
            image_store = view.image_store()
    else:
        data = prepared_table(resolved).lazy() if prepared else scan_tables(paths)
        manifest = resolved.manifest or {}
        provenance = manifest.get("provenance", {})
        if provenance.get("task") == "forecast":
            if source.action_semantics != "executed_predictions":
                raise ValueError(
                    "forecast sources require explicit action_semantics='executed_predictions'"
                )
            if provenance["prediction_columns"] != policy.action_columns:
                raise ValueError("Recorded predictions differ from policy actions")
            if (
                policy.previous_action_inputs or policy.return_conditioned
            ) and CONTEXT_COLUMN in data.collect_schema():
                data = data.drop(CONTEXT_COLUMN)
            data = data.with_columns(
                pl.struct(
                    [
                        pl.col("__prediction_" + name).alias(name)
                        for name in policy.action_columns
                    ]
                ).alias(ACTION_RECORD)
            )
        if {"sequenceId", "itemPosition"} - set(data.collect_schema().names()):
            raise ValueError("policy tables require sequenceId and itemPosition")
        table = IndexedTable(data, "sequenceId")
        metadata = resolved.metadata
        if (
            prepared
            and metadata
            and (metadata.image_sources or metadata.preconditioning_image)
        ):
            image_store = ImageStore(
                metadata.image_sources,
                metadata.preconditioning_image,
                str(resolved.project_root),
            )

    static_table = None
    if static_path:
        static_table = IndexedTable(scan_tables((Path(static_path),)), "sequenceId")
        if (
            any(end - start != 1 for start, end in static_table.ranges.values())
            or not table.ranges.keys() <= static_table.ranges.keys()
        ):
            raise ValueError(
                "static data must cover every selected sequence ID without duplicates"
            )
    return [
        TableEpisodeReference(
            source,
            policy,
            identity,
            table,
            metadata=metadata,
            prepared=prepared,
            static_table=static_table,
            image_store=image_store,
        )
        for identity in table.ranges
    ]
