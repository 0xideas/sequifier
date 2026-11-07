"""Shared categorical vocabulary and stable sample-statistics fitting."""

import math
import warnings
from typing import Union

import numpy as np
import polars as pl

from sequifier.special_tokens import SPECIAL_TOKEN_IDS
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


@beartype
def create_id_map(data: pl.DataFrame, column: str) -> dict[Union[str, int], int]:
    """Map sorted user values to IDs after reserved tokens."""
    ids = sorted(
        [int(x) if not isinstance(x, str) else x for x in np.unique(data[column])]
    )  # type: ignore

    if isinstance(ids[0], str):
        if SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.mask] in ids:
            raise ValueError(
                f"Found value '{SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.mask]}' in {column}, this is invalid"
            )

        for special_val in [
            SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.unknown],
            SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.other],
        ]:
            if special_val in ids:
                warnings.warn(
                    f"Found special value {special_val} in {column}, these will be combined with the sequifier-internal special value {special_val}"
                )
        ids = [
            id_
            for id_ in ids
            if id_
            not in [
                SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.unknown],
                SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.other],
            ]
        ]
        id_map = {id_: i + SPECIAL_TOKEN_IDS.user_start for i, id_ in enumerate(ids)}
        id_map[SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.unknown]] = (
            SPECIAL_TOKEN_IDS.unknown
        )
        id_map[SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.other]] = (
            SPECIAL_TOKEN_IDS.other
        )
    else:
        id_map = {id_: i + SPECIAL_TOKEN_IDS.user_start for i, id_ in enumerate(ids)}
    return dict(id_map)
