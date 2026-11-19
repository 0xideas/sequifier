"""Indexed table episode readers and canonical table-to-episode validation."""

import tempfile
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from sequifier.config.metadata import DatasetMetadata, load_dataset_metadata
from sequifier.config.policy_inputs import PolicyEpisodeSpec
from sequifier.config.policy_train_config import SequenceSourceConfig
from sequifier.io.images import ImageStore
from sequifier.io.preprocessing import denormalize_values
from sequifier.io.sequence import (
    SequenceEpisode,
    TransitionColumns,
    ordered_sequence_rows,
)
from sequifier.io.values import domain_scalar, numeric_scalar
from sequifier.trajectory.prepared import prepared_table

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
            scans.append(pl.scan_csv(path, infer_schema_length=None))
        elif path.suffix.lower() == ".parquet":
            scans.append(pl.scan_parquet(path))
        else:
            raise ValueError("table sources require CSV or Parquet files")
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
                if isinstance(identity, bool) or not isinstance(identity, (int, str)):
                    raise ValueError(
                        "sequence IDs must be non-null integers or strings"
                    )
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

    def __init__(self, view):
        self.view = view
        self.ranges = view.split_bounds()

    def load(self, identity):
        start, end = self.ranges[identity]
        return self.view.read(identity, start - 1, end).with_columns(
            (pl.col("itemPosition") >= start).alias("__decision")
        )


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
    sequence_id: str | int
    table: IndexedTable | IndexedEpisodeTable
    metadata: DatasetMetadata | None = None
    prepared: bool = False
    static_table: IndexedTable | None = None
    image_store: ImageStore | None = None

    @property
    def content_key(self):
        if isinstance(self.table, IndexedEpisodeTable):
            return self.table.view.dataset["sequences"][str(self.sequence_id)].get(
                "content_key"
            )
        return None

    def load(self) -> SequenceEpisode:
        data = ordered_sequence_rows(self.table.load(self.sequence_id))
        decode = model_decoder(
            self.metadata if self.prepared or self.source.encoding == "model" else None,
            self.policy.actions,
        )

        preceding_row = None
        if self.prepared:
            decisions = data.filter(pl.col("__decision"))
            if decisions.is_empty():
                raise ValueError("prepared episode contains no split-owned decisions")
            first = decisions["itemPosition"][0]
            previous = data.filter(pl.col("itemPosition") == first - 1)
            if len(previous):
                preceding_row = previous.row(0, named=True)
            data = decisions
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
            preceding_row=preceding_row,
            static_data=static_data,
            episode_end=self.source.episode_end,
            episode_namespace=self.source.episode_namespace,
            decode=decode,
        )


def table_episode(
    data,
    policy,
    *,
    sequence_id,
    preceding_row=None,
    static_data=None,
    episode_end="unknown",
    episode_namespace="",
    decode=domain_scalar,
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
            return canonical_action(row[ACTION_RECORD])
        return {name: decode(row[name], name) for name in action_names}

    def values(column, decoder=decode):
        return tuple(decoder(value, column) for value in data[column])

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
            (feature_value(value, name) for value in static_data[name]),
            f"static input {name}",
        )
        for name in static_inputs
    }
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
                    name: feature_value(value, name)
                    for name, value in zip(observations, entries, strict=True)
                }
                if all(present)
                else None
            )
    groups = optional_values(GROUP_COLUMN, domain_scalar)
    identity = (
        constant(values(EPISODE_ID_COLUMN, domain_scalar), "episode ID")
        if EPISODE_ID_COLUMN in data.columns
        else sequence_id
    )
    return SequenceEpisode(
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
            rewards=optional_values(
                REWARD_COLUMN,
                lambda value, column: numeric_scalar(decode(value, column), "reward"),
            ),
            successors=tuple(successors) if successors is not None else None,
        ),
    )


def table_episodes(source, policy, paths, *, prepared=False):
    from sequifier.io.indexed_sequences import IndexedView, is_indexed_view

    view = (
        IndexedView.load(paths[0])
        if prepared and len(paths) == 1 and is_indexed_view(paths[0])
        else None
    )
    if view is None:
        data = prepared_table(paths, source).lazy() if prepared else scan_tables(paths)
        if {"sequenceId", "itemPosition"} - set(data.collect_schema().names()):
            raise ValueError("policy tables require sequenceId and itemPosition")
        table = IndexedTable(data, "sequenceId")
    else:
        table = IndexedEpisodeTable(view)
    if set(source.columns.observations) - set(policy.observation_columns) or set(
        source.columns.static_inputs
    ) - set(policy.static_columns):
        raise ValueError("table columns must name declared policy features")
    static_table = None
    if source.static_data_path:
        static_table = IndexedTable(
            scan_tables((Path(source.static_data_path),)), "sequenceId"
        )
        if (
            any(end - start != 1 for start, end in static_table.ranges.values())
            or not table.ranges.keys() <= static_table.ranges.keys()
        ):
            raise ValueError(
                "static data must cover every selected sequence ID without duplicates"
            )
    metadata = (
        view.metadata()
        if view is not None
        else (
            load_dataset_metadata(source.metadata_config_path)
            if source.metadata_config_path
            else None
        )
    )
    image_store = None
    if (
        prepared
        and metadata is not None
        and (metadata.image_sources or metadata.preconditioning_image is not None)
    ):
        image_store = ImageStore(
            metadata.image_sources, metadata.preconditioning_image, source.project_root
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
