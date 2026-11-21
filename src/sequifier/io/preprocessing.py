"""Shared categorical vocabulary and stable sample-statistics fitting."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import polars as pl

from sequifier.helpers import polars_dtype_from_name
from sequifier.io.categorical import CategoricalCodec, validate_category_values
from sequifier.typechecking import beartype


@beartype
def _finite_mean_and_std(column: pl.Series, column_name: str) -> tuple[float, float]:
    """Compute sample statistics without overflowing intermediate squares."""
    values = column.to_numpy().astype(np.float64, copy=False)
    if values.size == 0:
        raise ValueError(f"Cannot compute statistics for empty column {column_name!r}.")
    if not np.isfinite(values).all():
        raise ValueError(f"Column {column_name!r} contains non-finite values.")

    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0, 0.0

    scaled_values = values / scale
    mean = float(np.mean(scaled_values) * scale)
    std = 0.0 if values.size <= 1 else float(np.std(scaled_values, ddof=1) * scale)
    if not math.isfinite(mean) or not math.isfinite(std):
        raise ValueError(
            f"Column {column_name!r} has a numeric range too large for finite "
            "normalization statistics."
        )
    return mean, std


@beartype
def get_combined_statistics(
    n1: int, mean1: float, std1: float, n2: int, mean2: float, std2: float
) -> tuple[float, float]:
    """Combine two mean/std summaries."""
    if n1 == 0:
        return mean2, std2
    if n2 == 0:
        return mean1, std1

    scale = max(abs(mean1), abs(mean2), abs(std1), abs(std2))
    if not math.isfinite(scale):
        raise ValueError("Cannot combine non-finite normalization statistics.")
    if scale == 0.0:
        return 0.0, 0.0

    scaled_mean1 = mean1 / scale
    scaled_mean2 = mean2 / scale
    scaled_std1 = std1 / scale
    scaled_std2 = std2 / scale
    combined_mean_scaled = (n1 * scaled_mean1 + n2 * scaled_mean2) / (n1 + n2)
    combined_mean = combined_mean_scaled * scale

    if n1 + n2 <= 1:
        return combined_mean, 0.0

    sum_of_squares1 = (n1 - 1) * scaled_std1**2 + n1 * (
        scaled_mean1 - combined_mean_scaled
    ) ** 2
    sum_of_squares2 = (n2 - 1) * scaled_std2**2 + n2 * (
        scaled_mean2 - combined_mean_scaled
    ) ** 2

    combined_std = (
        math.sqrt((sum_of_squares1 + sum_of_squares2) / (n1 + n2 - 1)) * scale
    )
    if not math.isfinite(combined_mean) or not math.isfinite(combined_std):
        raise ValueError(
            "Combined numeric range is too large for finite normalization statistics."
        )

    return combined_mean, combined_std


NORMALIZATION_EPSILON = 1e-9


def normalization_statistics(statistics: Mapping) -> dict[str, dict[str, float]]:
    """Validate persisted statistics once, independently of the array backend."""
    result = {}
    for column, values in statistics.items():
        mean, std = values.get("mean"), values.get("std")
        if (
            isinstance(mean, bool)
            or isinstance(std, bool)
            or not isinstance(mean, (int, float))
            or not isinstance(std, (int, float))
            or not math.isfinite(mean)
            or not math.isfinite(std)
            or std < 0
        ):
            raise ValueError(
                f"{column}: normalization requires finite means and nonnegative scales"
            )
        result[column] = {"mean": float(mean), "std": float(std)}
    return result


def normalize_values(values, statistics):
    """Standardize scalar, NumPy, Torch, or Polars values with fixed statistics."""
    return (values - statistics["mean"]) / (statistics["std"] + NORMALIZATION_EPSILON)


def denormalize_values(values, statistics):
    return values * (statistics["std"] + NORMALIZATION_EPSILON) + statistics["mean"]


def normalize_input_values(
    values, statistics: Mapping[str, float], column_type: str | None
):
    """Use the same floating-point operations for prepared and online values."""
    precision = (column_type or "").lower()
    reciprocal = 1.0 / (statistics["std"] + NORMALIZATION_EPSILON)
    if isinstance(values, pl.Expr):
        dtype = {"float16": pl.Float16, "float32": pl.Float32}.get(
            precision, pl.Float64
        )
        arithmetic = pl.Float32 if precision == "float16" else dtype
        values = values.cast(dtype).cast(arithmetic)
        return (
            (values - pl.lit(statistics["mean"], dtype=arithmetic))
            * pl.lit(reciprocal, dtype=arithmetic)
        ).cast(dtype)
    if precision == "float16":
        values = np.asarray(values, dtype=np.float16).astype(np.float32)
        result = (values - np.float32(statistics["mean"])) * np.float32(reciprocal)
        return result.astype(np.float16)
    dtype = np.float32 if precision == "float32" else np.float64
    return (np.asarray(values, dtype=dtype) - dtype(statistics["mean"])) * dtype(
        reciprocal
    )


@dataclass(frozen=True)
class FittedFeatureTransform:
    """A fitted column's encoding and stored precision, independent of its row role."""

    dtype: str
    codec: CategoricalCodec | None
    statistics: Mapping | None

    @staticmethod
    def inferred_dtype(name, raw_dtype, maps, stats):
        if name in maps:
            return "Int64"
        if name in stats and not str(raw_dtype).startswith("Float"):
            return "Float64"
        return str(raw_dtype)

    @property
    def output_dtype(self):
        if self.dtype.startswith(("Date", "Time", "Duration")):
            return pl.Int64
        return polars_dtype_from_name(self.dtype)

    def expression(self, source: str) -> pl.Expr:
        if self.codec is not None:
            return self.codec.expression(source)
        if self.statistics is not None:
            return cast(
                pl.Expr,
                normalize_input_values(pl.col(source), self.statistics, self.dtype),
            )
        return pl.col(source)

    def stored_expression(self, column: pl.Series, *, preserve_nulls=False) -> pl.Expr:
        if self.codec is not None and self.codec.cardinality:
            validate_category_values(column, hashing=self.codec.hashing is not None)
        value = self.expression(column.name)
        if preserve_nulls:
            value = pl.when(pl.col(column.name).is_not_null()).then(value)
        return value.cast(self.output_dtype).alias(column.name)
