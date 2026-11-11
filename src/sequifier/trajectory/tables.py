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
    table: IndexedTable
    metadata: DatasetMetadata | None = None
    prepared: bool = False
    static_table: IndexedTable | None = None
    image_store: ImageStore | None = None

    def load(self) -> SequenceEpisode:
        columns = self.source.columns
        data = ordered_sequence_rows(
            self.table.load(self.sequence_id), columns.item_position
        )
        decode = model_decoder(
            self.metadata if self.prepared or self.source.encoding == "model" else None,
            self.policy.actions,
        )

        preceding = None
        if self.prepared:
            decisions = data.filter(pl.col("__decision"))
            if decisions.is_empty():
                raise ValueError("prepared episode contains no split-owned decisions")
            first = decisions[columns.item_position][0]
            previous = data.filter(pl.col(columns.item_position) == first - 1)
            if len(previous):
                mapping = columns.actions or {
                    name: name for name in self.policy.action_columns
                }
                preceding = {
                    name: decode(previous[column][0], name)
                    for name, column in mapping.items()
                }
            data = decisions
        static_data = None
        if self.static_table is not None:
            static_data = self.static_table.load(self.sequence_id)
        if self.image_store is not None:
            import torch

            images, masks = self.image_store.gather(
                torch.tensor([int(self.sequence_id)]),
                torch.tensor([data[columns.item_position].to_list()]),
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
            columns,
            sequence_id=self.sequence_id,
            preceding=preceding,
            static_data=static_data,
            episode_end=self.source.episode_end,
            episode_namespace=self.source.episode_namespace,
            decode=decode,
        )


def table_features(data, policy, columns, static_data):
    observations, static_inputs = (
        {name: mapping.get(name, name) for name in names}
        for names, mapping in (
            (policy.observation_columns, columns.observations),
            (policy.static_columns, columns.static_inputs),
        )
    )
    action_column = columns.action
    action_columns = columns.actions or (
        {name: name for name in policy.action_columns}
        if columns.action is None
        else {name: action_column for name in policy.action_columns}
    )
    inputs = set(observations.values())
    if static_data is None:
        inputs.update(static_inputs.values())
    if (
        set(action_columns.values()) & inputs
        or columns.previous_action in action_columns.values()
    ):
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
        | {columns.item_position}
        | set(action_columns.values())
        | set(columns.previous_actions.values())
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

    return observations, static_inputs


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
    if decode is domain_scalar:

        def identity_decode(value, column):
            return value

        decode = identity_decode

    if data.is_empty():
        raise ValueError("sequence source changed while loading")
    action_column = columns.action
    action_columns = columns.actions or (
        {name: name for name in policy.action_columns}
        if columns.action is None
        else {name: action_column for name in policy.action_columns}
    )
    observations, static_inputs = table_features(data, policy, columns, static_data)

    def canonical_action(value):
        if value is None:
            return value
        return {
            name: entry
            for name, entry in value.items()
            if name in policy.actions or entry is not None
        }

    def values(column, decoder=decode):
        if policy.input_encoding == "model" and column in observations.values():
            return tuple(data[column].to_list())
        return tuple(decoder(value, column) for value in data[column])

    def optional_values(column, decoder=decode):
        if (
            column is None
            or column not in data.columns
            or data[column].null_count() == len(data)
        ):
            return None
        return values(column, decoder)

    def constant(entries, label):
        entries = list(entries)
        if not entries or any(value != entries[0] for value in entries[1:]):
            raise ValueError(f"{label} must remain constant within an episode")
        return entries[0]

    positions = values(columns.item_position, lambda value, _: value)
    if set(action_columns) != set(policy.actions):
        raise ValueError("table action mappings must cover every action field")
    actions = tuple(
        canonical_action(dict(row[columns.action]))
        if columns.action is not None
        and not columns.actions
        and isinstance(row[columns.action], dict)
        else {
            name: decode(row[column], name) for name, column in action_columns.items()
        }
        for row in data.iter_rows(named=True)
    )
    if columns.previous_actions:
        if set(columns.previous_actions) != set(policy.actions):
            raise ValueError("previous action mappings must cover every action field")
        previous = tuple(
            {name: row[column] for name, column in columns.previous_actions.items()}
            for row in data.iter_rows(named=True)
        )
        preceding = (
            None
            if all(value is None for value in previous[0].values())
            else previous[0]
        )
        if previous[1:] != actions[:-1]:
            raise ValueError(
                "previous actions must lag recorded actions by one decision"
            )
    if columns.previous_action is not None:
        preceding_actions = values(
            columns.previous_action,
            lambda value, _: None if value is None else (canonical_action(value)),
        )
        if preceding_actions[1:] != actions[:-1]:
            raise ValueError(
                "previous-action column must lag executed actions by exactly one decision"
            )
        preceding = preceding_actions[0]
    if static_data is not None:
        if len(static_data) != 1:
            raise ValueError("static data requires exactly one row per sequenceId")
    else:
        static_data = data
    static = {
        name: constant(
            (
                value if policy.input_encoding == "model" else decode(value, column)
                for value in static_data[column]
            ),
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
        transitions[declared] = (False,) * (len(data) - 1) + (True,)
    successors = None
    if columns.successors:
        if columns.successors.keys() != observations.keys():
            raise ValueError("successor mapping must cover every observation column")
        successors = []
        for row in data.iter_rows(named=True):
            present = [
                row[column] is not None for column in columns.successors.values()
            ]
            if any(present) and not all(present):
                raise ValueError("a missing successor must mask all observation fields")
            successors.append(
                {
                    name: row[column]
                    if policy.input_encoding == "model"
                    else decode(row[column], observations[name])
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


def table_episodes(source, policy, paths, *, prepared=False):
    data = prepared_table(paths, source).lazy() if prepared else scan_tables(paths)
    columns = source.columns
    required = {columns.sequence_id, columns.item_position}
    if required - set(data.collect_schema().names()):
        raise ValueError(
            "policy tables require sequenceId and itemPosition, or explicit column mappings"
        )
    table = IndexedTable(data, columns.sequence_id)
    static_table = None
    if source.static_data_path:
        static_table = IndexedTable(
            scan_tables((Path(source.static_data_path),)), columns.sequence_id
        )
        if (
            any(end - start != 1 for start, end in static_table.ranges.values())
            or not table.ranges.keys() <= static_table.ranges.keys()
        ):
            raise ValueError(
                "static data must cover every selected sequence ID without duplicates"
            )
    metadata = (
        load_dataset_metadata(source.metadata_config_path)
        if source.metadata_config_path
        else None
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
            metadata,
            prepared,
            static_table,
            image_store,
        )
        for identity in table.ranges
    ]
