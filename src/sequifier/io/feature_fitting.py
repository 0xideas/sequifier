"""Mergeable feature statistics for tabular data and episode views."""

import warnings
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Union

import polars as pl

from sequifier.io.categorical import HASH_BUCKET_LABEL_PREFIX
from sequifier.io.preprocessing import _finite_mean_and_std, get_combined_statistics
from sequifier.special_tokens import SPECIAL_TOKEN_IDS, SPECIAL_TOKEN_LABELS
from sequifier.typechecking import beartype


@beartype
def mask_expression(series: Any, mask_dtype: Any, mask_column: str) -> pl.Expr:
    mask_col = pl.col(mask_column)

    if mask_dtype == pl.Boolean:
        return mask_col

    if mask_dtype.is_numeric():
        if not series.drop_nulls().is_in([0, 1]).all():
            raise ValueError(
                f"Mask column {mask_column} contains inadmissible values not in (0, 1)"
            )
        return mask_col == 1

    raise ValueError(
        f"Column {mask_column} must be boolean or numeric, got {mask_dtype}"
    )


@beartype
def _category_sort_key(value: Any) -> tuple[str, Any]:
    """Provide deterministic ordering, including for defensive mixed-type input."""
    if isinstance(value, (str, int, float, bool)):
        return type(value).__name__, value
    return type(value).__name__, repr(value)


def _reserved_id_map() -> dict[str | int, int]:
    return {
        label: code
        for label, code in SPECIAL_TOKEN_IDS.ids_by_label.items()
        if code != SPECIAL_TOKEN_IDS.mask
    }


def create_id_map(values, column: str) -> dict[str | int, int]:
    """Assign stable IDs to the distinct labels observed across all fitting inputs."""
    if not values:
        raise ValueError(f"cannot fit empty categorical column {column!r}")
    strings = {isinstance(value, str) for value in values}
    if len(strings) > 1:
        raise ValueError(
            f"{column}: mixed string/numeric category labels are ambiguous"
        )
    mapping: dict[str | int, int] = {}
    if True in strings:
        labels = set(values)
        mask = SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.mask]
        if mask in labels:
            raise ValueError(f"Found value {mask!r} in {column}, this is invalid")
        mapping.update(_reserved_id_map())
        for label in mapping.keys() & labels:
            warnings.warn(
                f"Found special value {label} in {column}, using its reserved ID"
            )
        labels.difference_update(mapping)
    else:
        if any(value is None or value != int(value) for value in values):
            raise ValueError(f"{column}: categorical numeric labels must be integers")
        labels = {int(value) for value in values}
    mapping.update(
        (label, index)
        for index, label in enumerate(sorted(labels), SPECIAL_TOKEN_IDS.user_start)
    )
    return mapping


@beartype
def _finalize_cardinality_maps(
    id_maps: dict[str, dict[Union[str, int], int]],
    value_counts: dict[str, Counter],
    cardinality_config: Mapping[str, dict[str, Any]],
) -> dict[str, dict[Union[str, int], int]]:
    """Build retained-value maps followed by optional hashing buckets."""
    for column, config in cardinality_config.items():
        retained = []
        if "min_count" in config or "max_categories" in config:
            counts = value_counts.get(column, Counter())
            if not counts:
                raise ValueError(
                    "No unmasked examples found for cardinality-controlled column "
                    f"{column!r}"
                )
            user_counts = {
                value: count
                for value, count in counts.items()
                if value not in SPECIAL_TOKEN_LABELS
            }
            if "min_count" in config:
                retained = [
                    value
                    for value, count in user_counts.items()
                    if count >= int(config["min_count"])
                ]
            else:
                retained = [
                    value
                    for value, _ in sorted(
                        user_counts.items(),
                        key=lambda item: (-item[1], _category_sort_key(item[0])),
                    )[: int(config["max_categories"])]
                ]
        mapping = create_id_map(retained, column) if retained else _reserved_id_map()
        hashing = config.get("hashing")
        if hashing is not None:
            start = SPECIAL_TOKEN_IDS.user_start + len(retained)
            mapping.update(_reserved_id_map())
            mapping.update(
                {
                    f"{HASH_BUCKET_LABEL_PREFIX}{index}]": start + index
                    for index in range(int(hashing["num_buckets"]))
                }
            )
        id_maps[column] = mapping
    return id_maps


@dataclass(frozen=True)
class FittingRules:
    precomputed: dict = field(default_factory=dict)
    cardinality: dict = field(default_factory=dict)
    categorical: list[str] | None = None
    real: list[str] | None = None


@dataclass
class FeatureStatistics:
    moments: dict[str, dict[str, float]] = field(default_factory=dict)
    vocabularies: dict[str, set] = field(default_factory=dict)
    counts: dict[str, Counter] = field(default_factory=dict)

    def _merge_moments(self, column, count, mean, std):
        previous = self.moments.get(column)
        if previous is not None:
            old_count = int(previous["count"])
            mean, std = get_combined_statistics(
                count, mean, std, old_count, previous["mean"], previous["std"]
            )
            count += old_count
        self.moments[column] = {"count": float(count), "mean": mean, "std": std}

    def observe_real(self, values):
        mean, std = _finite_mean_and_std(values, values.name)
        self._merge_moments(values.name, len(values), mean, std)

    def observe(self, data, columns, rules, *, mask_column=None):
        if mask_column is not None and mask_column in data.columns:
            data = data.filter(
                ~mask_expression(
                    data[mask_column], data.schema[mask_column], mask_column
                )
            )
        if data.is_empty():
            return
        for name in columns:
            values = data[name]
            dtype = values.dtype
            categorical = dtype.is_integer() or dtype in (
                pl.String,
                pl.Object,
                pl.Categorical,
                pl.Boolean,
            )
            if name in (rules.categorical or ()) or (
                categorical and name not in (rules.real or ())
            ):
                if not categorical:
                    raise ValueError(f"{name}: unsupported categorical dtype {dtype}")
                if name in rules.precomputed:
                    continue
                limit = rules.cardinality.get(name)
                if limit is None:
                    self.vocabularies.setdefault(name, set()).update(
                        values.unique().to_list()
                    )
                elif "min_count" in limit or "max_categories" in limit:
                    self.counts.setdefault(name, Counter()).update(
                        values.drop_nulls().to_list()
                    )
            elif name in rules.cardinality:
                raise ValueError(
                    f"cardinality_config column {name!r} is not categorical"
                )
            elif name in (rules.real or ()) or dtype.is_float():
                if not dtype.is_numeric() or name in rules.precomputed:
                    raise ValueError(
                        f"{name}: real columns require numeric values and cannot have a precomputed map"
                    )
                self.observe_real(values)
            elif not dtype.is_temporal():
                raise ValueError(f"Column {name} has unsupported dtype: {dtype}")

    def merge(self, other):
        for name, values in other.vocabularies.items():
            self.vocabularies.setdefault(name, set()).update(values)
        for name, counts in other.counts.items():
            self.counts.setdefault(name, Counter()).update(counts)
        for name, chunk in other.moments.items():
            self._merge_moments(name, int(chunk["count"]), chunk["mean"], chunk["std"])

    def finish(self, rules):
        maps = dict(rules.precomputed)
        for name, values in self.vocabularies.items():
            maps[name] = create_id_map(values, name)
        return _finalize_cardinality_maps(
            maps, self.counts, rules.cardinality
        ), self.moments
