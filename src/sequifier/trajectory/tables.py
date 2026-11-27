"""Indexed table episode readers and canonical table-to-episode validation."""

from __future__ import annotations

import tempfile
from bisect import bisect_right
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, cast

import polars as pl

from sequifier.config.metadata import DatasetMetadata
from sequifier.config.policy_inputs import PolicyEpisodeSpec
from sequifier.config.policy_train_config import SequenceSourceConfig
from sequifier.io.images import ImageStore
from sequifier.io.preprocessing import denormalize_values
from sequifier.io.sequence import (
    CONTEXT_RECORDS,
    DOMAIN_OBSERVATIONS,
    DOMAIN_STATIC,
    ContextObservation,
    SequenceEpisode,
    TransitionColumns,
    episode_identity,
    ordered_sequence_rows,
    source_fields,
)
from sequifier.io.values import domain_scalar, numeric_scalar
from sequifier.trajectory.prepared import prepared_table
from sequifier.trajectory.row_layout import ACTION_METADATA

if TYPE_CHECKING:
    from sequifier.io.indexed_sequences import IndexedImageStore
    from sequifier.trajectory.sources import ResolvedSequenceSource

GROUP_COLUMN = "groupId"
SEQUENCE_NAME_COLUMN = "sequenceName"
REWARD_COLUMN = "reward"
TRANSITION_FLAGS = ("terminated", "truncated")


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
    return pl.concat(scans, how="vertical_relaxed")


class IndexedTable:
    """Canonical sorted storage with direct row-group access per episode.

    The temporary directory lives as long as its episode references. CSV parsing
    and sorting happen once; cache misses never filter the original source.
    """

    def __init__(self, data, sequence_column, *, bounds=None):
        import pyarrow.parquet as pq

        self.directory = tempfile.TemporaryDirectory(prefix="sequifier-episodes-")
        self.path = Path(self.directory.name) / "episodes.parquet"
        order = (
            [sequence_column, "itemPosition"]
            if "itemPosition" in data.collect_schema()
            else [sequence_column]
        )
        data.sort(order).sink_parquet(self.path, row_group_size=8192)
        table = pq.ParquetFile(self.path)
        self.offsets = [0]
        for i in range(table.num_row_groups):
            self.offsets.append(self.offsets[-1] + table.metadata.row_group(i).num_rows)
        self.ranges = {}
        self.bounds = bounds
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

    @cached_property
    def split_content_keys(self):
        from sequifier.io.partitioning import table_content_keys

        return table_content_keys(pl.scan_parquet(self.path))

    def load(self, identity):
        import pyarrow.parquet as pq

        start, end = self.ranges[identity]
        first = bisect_right(self.offsets, start) - 1
        last = bisect_right(self.offsets, end - 1) - 1
        with pq.ParquetFile(self.path) as table:
            rows = table.read_row_groups(list(range(first, last + 1)))
        return cast(
            pl.DataFrame,
            pl.from_arrow(rows.slice(start - self.offsets[first], end - start)),
        )


class IndexedEpisodeTable:
    """Read split-owned episodes directly from the committed sequence store."""

    def __init__(self, view, context_length):
        self.context_length = context_length
        self.view = view
        self.bounds = view.split_bounds()
        self.ranges = {
            identity: (low, high)
            for identity, (low, high) in self.bounds.items()
            if high > max(low, view.dataset["sequences"][str(identity)]["position"] + 1)
        }
        if not self.ranges:
            raise ValueError(
                f"{view.path}: policy episodes require recorded split membership"
            )

    def load(self, identity):
        start, end = self.ranges[identity]
        history = self.view.history_start(
            identity, start, end, start - self.context_length
        )
        record = self.view.dataset["sequences"][str(identity)]
        rows = self.view.read(
            identity, max(record["position"], min(history, start - 1)), end
        )
        for field, column in (
            ("episode_id", SEQUENCE_NAME_COLUMN),
            ("group_id", GROUP_COLUMN),
        ):
            if record.get(field) is not None and column not in rows.columns:
                rows = rows.with_columns(pl.lit(record[field]).alias(column))
        return rows


def observation_mask(frame, raw_column, *, recorded_column=None):
    """Combine the available recorded and configured row masks in boolean form."""
    from sequifier.io.feature_fitting import mask_expression

    schema = frame.collect_schema()
    columns = [
        name
        for name in dict.fromkeys((recorded_column, raw_column))
        if name is not None and name in schema
    ]
    if raw_column and not columns:
        raise ValueError(f"Missing observation mask column {raw_column!r}")
    result = pl.repeat(False, pl.len())
    for name in columns:
        if isinstance(frame, pl.LazyFrame):
            values = frame.select(pl.col(name).unique()).collect().to_series()
        else:
            values = frame[name]
        if values.null_count():
            raise ValueError(f"Mask column {name} contains null values")
        result = result | mask_expression(values, schema[name], name)
    return result


def domain_action_columns(rows, action_columns, observations, static):
    raw = rows.schema.get(DOMAIN_OBSERVATIONS)
    if not isinstance(raw, pl.Struct):
        return set()
    return (set(action_columns) & set(rows.columns) & raw.to_schema().keys()) - {
        *observations,
        *static,
    }


def project_prepared_rows(
    rows,
    metadata,
    observations,
    static,
    actions,
    *,
    physical_inputs,
    action_mapping=None,
):
    """Recover raw action labels and optionally project raw observations."""
    action_columns = action_mapping.values() if action_mapping else actions
    rows = rows.with_columns(
        pl.col(DOMAIN_OBSERVATIONS).struct.field(column).alias(column)
        for column in domain_action_columns(rows, action_columns, observations, static)
    )
    if physical_inputs:
        encoded = set(metadata.column_data_types)
        rows = rows.with_columns(
            *source_fields(
                rows, DOMAIN_OBSERVATIONS, [c for c in observations if c in encoded]
            ),
            *source_fields(rows, DOMAIN_STATIC, [c for c in static if c in encoded]),
        )
    return rows


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
    action_mapping: dict | None = None
    mask_column: str | None = None
    recorded_mask_column: str | None = None
    seed: dict | None = None

    @property
    def content_key(self):
        if isinstance(self.table, IndexedEpisodeTable):
            return self.table.view.dataset["sequences"][str(self.sequence_id)].get(
                "content_key"
            )
        return None

    @property
    def split_content_key(self):
        if self.prepared or not isinstance(self.table, IndexedTable):
            return None
        key = self.table.split_content_keys[self.sequence_id]
        if self.static_table is not None:
            return key, self.static_table.split_content_keys[self.sequence_id]
        return key

    @property
    def used_for_fitting(self) -> bool:
        return (
            isinstance(self.table, IndexedEpisodeTable)
            and self.sequence_id in self.table.view.fitted_sequences
        )

    def source_columns(self, positions, columns):
        """Read auxiliary fields at the same contiguous observation coordinates."""
        rows = self.table.load(self.sequence_id)
        if missing := set(columns) - set(rows.columns):
            raise ValueError(f"Missing recorded columns: {sorted(missing)}")
        offset = positions[0] - rows["itemPosition"][0]
        if offset < 0 or offset + len(positions) > len(rows):
            raise ValueError("Source columns do not cover the requested observations")
        return rows.slice(offset, len(positions)).select(columns)

    def load(self) -> SequenceEpisode:
        data = ordered_sequence_rows(self.table.load(self.sequence_id))
        decode = model_decoder(
            self.metadata if self.prepared or self.source.encoding == "model" else None,
            self.policy.actions,
        )

        owned_start = None
        seed = self.seed or {}
        if isinstance(self.table, IndexedEpisodeTable):
            view = self.table.view
            record = view.dataset["sequences"][str(self.sequence_id)]
            seed = record.get("trajectory") or view.dataset.get("provenance", {}).get(
                "sequence_seeds", {}
            ).get(str(self.sequence_id), seed)
            if data["itemPosition"][0] != record["position"]:
                seed = {}
        context = ()
        if payload := seed.get("context"):
            if self.prepared and self.policy.input_encoding == "model":
                from sequifier.io.prepare_inference import encode_seed_context

                payload = encode_seed_context(
                    payload, self.metadata, self.policy.observation_columns
                )
            context = CONTEXT_RECORDS.validate_json(payload)
        if self.prepared:
            assert self.table.bounds is not None
            low, high = self.table.bounds[self.sequence_id]
            owned = data.filter(pl.col("itemPosition").is_between(low, high - 1))
            if owned.is_empty():
                raise ValueError("prepared episode contains no split-owned decisions")
            owned_start = max(low - 1, data["itemPosition"][0])
            if low > 0 and data["itemPosition"][0] >= low:
                raise ValueError(
                    "Split lacks the observation preceding its first target"
                )
            if not isinstance(self.table, IndexedEpisodeTable):
                data = data.filter(pl.col("itemPosition") < high)
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
        raw_columns = set(self.metadata.indexed_passthrough) if self.metadata else set()
        if self.prepared:
            physical = self.policy.input_encoding == "raw"
            if static_data is not None:
                data = data.join(static_data, on="sequenceId", validate="m:1")
                static_data = None
            data = project_prepared_rows(
                data,
                self.metadata,
                self.policy.observation_columns,
                self.policy.static_columns,
                self.policy.actions,
                physical_inputs=physical,
                action_mapping=self.action_mapping,
            )
            action_columns = (
                self.action_mapping.values()
                if self.action_mapping
                else self.policy.actions
            )
            raw_columns.update(
                domain_action_columns(
                    data,
                    action_columns,
                    self.policy.observation_columns,
                    self.policy.static_columns,
                )
            )
            if physical:
                raw_columns.update(
                    self.policy.observation_columns + self.policy.static_columns
                )
        return table_episode(
            data,
            self.policy,
            sequence_id=self.sequence_id,
            static_data=static_data,
            owned_start=owned_start,
            episode_end=self.source.episode_end,
            episode_namespace=self.source.episode_namespace,
            decode=decode,
            raw_columns=raw_columns,
            action_mapping=self.action_mapping,
            mask_column=self.mask_column
            or (self.metadata.raw_mask_column if self.metadata else None),
            recorded_mask_column=self.recorded_mask_column,
            seed_context=context,
            ordered=True,
            allow_fragment=self.prepared,
        )


def table_episode(
    data,
    policy,
    *,
    sequence_id,
    owned_start=None,
    static_data=None,
    episode_end="unknown",
    episode_namespace="",
    decode=domain_scalar,
    raw_columns=(),
    action_mapping=None,
    mask_column=None,
    recorded_mask_column=None,
    seed_context=(),
    ordered=False,
    allow_fragment=False,
) -> SequenceEpisode:
    """Read an episode using the policy's canonical feature names."""
    if not ordered:
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
    if data["itemPosition"][0] != 0 and not allow_fragment:
        raise ValueError("Trajectory tables must start at itemPosition zero")
    action_fields = action_mapping or {name: name for name in action_names}
    if set(action_fields) != set(action_names) or len(
        set(action_fields.values())
    ) != len(action_fields):
        raise ValueError("Recorded action columns differ from policy actions")
    action_columns = set(action_fields.values())
    inputs = set(observations) | (set(static_inputs) if static_data is None else set())
    if action_columns & inputs:
        raise ValueError("Action and observation columns must be distinct")
    if missing := (inputs | action_columns | {"itemPosition"}) - set(data.columns):
        raise ValueError(f"policy table is missing columns: {sorted(missing)}")
    masks = (
        data.select(
            observation_mask(
                data,
                mask_column or policy.raw_mask_column,
                recorded_column=recorded_mask_column,
            )
        )
        .to_series()
        .to_list()
    )
    present = data.select(pl.col(list(action_columns)).is_not_null())
    complete = present.select(pl.all_horizontal(pl.all())).to_series()
    if (
        len(data) < 2
        or not complete[1:].all()
        or ((data["itemPosition"][0] > 0 or any(present.row(0))) and not complete[0])
    ):
        raise ValueError(
            "Every arrival requires a complete action and a preceding observation"
        )
    if data["itemPosition"][0] == 0 and any(
        data[name][0] is not None
        for name in (*TRANSITION_FLAGS, REWARD_COLUMN)
        if name in data.columns
    ):
        raise ValueError("Initial observation cannot carry a transition outcome")

    def action_at(row):
        return {
            **{
                name: value
                for name, value in (row.get(ACTION_METADATA) or {}).items()
                if value is not None
            },
            **{
                name: row[column]
                if column in raw_columns
                else decode(row[column], column)
                for name, column in action_fields.items()
            },
        }

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

    first_position = data["itemPosition"][0]
    incoming = tuple(
        action_at(row) if i or complete[0] else None
        for i, row in enumerate(data.iter_rows(named=True))
    )
    if static_data is not None:
        if len(static_data) != 1:
            raise ValueError("static data requires exactly one row per sequenceId")
        if missing := set(static_inputs) - set(static_data.columns):
            raise ValueError(f"static table is missing columns: {sorted(missing)}")
    else:
        static_data = data

    def feature_value(value, name):
        return (
            value
            if policy.input_encoding == "model" or name in raw_columns
            else decode(value, name)
        )

    static = {
        name: constant(
            (feature_value(value, name) for value in static_data[name].to_list()),
            f"static input {name}",
        )
        for name in static_inputs
    }
    groups = optional_values(GROUP_COLUMN, domain_scalar)
    identity = (
        constant(data[SEQUENCE_NAME_COLUMN], "episode ID")
        if SEQUENCE_NAME_COLUMN in data.columns
        and data[SEQUENCE_NAME_COLUMN].null_count() < len(data)
        else None
    )

    def reward_value(value, column):
        return numeric_scalar(
            value if column in raw_columns else decode(value, column), "reward"
        )

    if any(
        name in data.columns and data[name].head(-1).any() for name in TRANSITION_FLAGS
    ):
        raise ValueError("A trajectory cannot continue after a boundary")
    context = seed_context
    if owned_start is not None and owned_start > first_position:
        offset = owned_start - first_position
        prefix = data.head(offset)
        first = max(0, offset - policy.context_length + 1)
        frames = []
        for i, row in enumerate(prefix.slice(first).iter_rows(named=True), first):
            frames.append(
                ContextObservation(
                    observation={
                        name: feature_value(row[name], name) for name in observations
                    },
                    masked=masks[i],
                    previous_action=incoming[i],
                    reward=reward_value(data[REWARD_COLUMN][i + 1], REWARD_COLUMN)
                    if REWARD_COLUMN in data.columns
                    and data[REWARD_COLUMN][i + 1] is not None
                    else None,
                )
            )
        context = (
            (*context, *frames)[-(policy.context_length - 1) :]
            if policy.context_length > 1
            else ()
        )
        incoming = incoming[offset:]
        first_position += offset
        data = data.slice(offset)
        masks = masks[offset:]

    # Strict boolean and length checks belong to TransitionColumns/SequenceEpisode.
    observation_values = {name: values(name, feature_value) for name in observations}
    data = data.slice(1)
    transitions = {
        field: optional_values(field, lambda value, _: value)
        for field in TRANSITION_FLAGS
    }
    declared = episode_end
    if declared != "unknown":
        current = transitions.get(declared)
        if current is not None and not current[-1]:
            raise ValueError("episode_end conflicts with the recorded final boundary")
        transitions[declared] = (False,) * (len(data) - 1) + (True,)
    return SequenceEpisode(
        sequence_id=int(sequence_id),
        episode_id=episode_namespace + episode_identity(sequence_id, identity),
        context=context,
        first_position=first_position,
        observation_masks=tuple(masks),
        observations=observation_values,
        incoming_actions=incoming,
        collection=tuple(data["__collection"].to_list())
        if "__collection" in data.columns
        else None,
        static_inputs=static,
        group_id=str(constant(groups, "collection group"))
        if groups is not None
        else None,
        transitions=TransitionColumns(
            **transitions,
            rewards=optional_values(REWARD_COLUMN, reward_value),
        ),
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
    static_path, image_store = resolved.static_path, resolved.image_store
    manifest = resolved.manifest or {}
    provenance = (
        (view.dataset if view else manifest).get("provenance")
        or getattr(resolved.metadata, "provenance", None)
        or {}
    )
    action_mapping = manifest.get("action_columns") or provenance.get("action_columns")
    sequence_seeds = provenance.get("sequence_seeds", {})
    policy.validate_source_units(provenance.get("policy_binding", {}).get("units"))
    policy.validate_source_units(provenance.get("source", {}).get("units"))
    if (
        provenance.get("task") == "forecast"
        and source.action_semantics != "executed_predictions"
    ):
        raise ValueError(
            "Forecast sources require action_semantics='executed_predictions'"
        )

    if source.action_columns:
        action_mapping = {
            name: name for name in policy.action_columns
        } | source.action_columns
    mask_column = source.mask_column
    recorded_mask_column = manifest.get("mask_column")
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
        bounds = None
        if prepared:
            rows, bounds = prepared_table(resolved)
            data = rows.lazy()
        else:
            data = scan_tables(paths)
            if manifest.get("sequence_metadata"):
                identities = pl.scan_parquet(static_path).select(
                    "sequenceId", SEQUENCE_NAME_COLUMN, GROUP_COLUMN
                )
                data = data.join(identities, on="sequenceId", validate="m:1")
        if {"sequenceId", "itemPosition"} - set(data.collect_schema().names()):
            raise ValueError("policy tables require sequenceId and itemPosition")
        table = IndexedTable(data, "sequenceId", bounds=bounds)
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
            action_mapping=action_mapping,
            mask_column=mask_column,
            recorded_mask_column=recorded_mask_column,
            seed=sequence_seeds.get(str(identity)),
        )
        for identity in table.ranges
    ]
