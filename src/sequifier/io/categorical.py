"""Shared categorical lookup rules for preprocessing and policy histories."""

import numpy as np
import polars as pl

from sequifier.special_tokens import SPECIAL_TOKEN_IDS, SPECIAL_TOKEN_LABELS

HASH_BUCKET_LABEL_PREFIX = "[hash_bucket:"
_MASK = (1 << 64) - 1


def is_hash_bucket_label(value):
    return (
        isinstance(value, str)
        and value.startswith(HASH_BUCKET_LABEL_PREFIX)
        and value.endswith("]")
    )


def validate_category_values(column: pl.Series, *, hashing: bool) -> None:
    if column.dtype not in (pl.String, pl.Categorical):
        return
    values = column.drop_nulls()
    mask = SPECIAL_TOKEN_IDS.labels_by_id[SPECIAL_TOKEN_IDS.mask]
    if values.eq(mask).any():
        raise ValueError(f"Found value {mask!r} in {column.name}, this is invalid")
    if hashing and any(is_hash_bucket_label(value) for value in values.to_list()):
        raise ValueError(
            f"Values beginning with {HASH_BUCKET_LABEL_PREFIX!r} are reserved for hashing buckets in {column.name}"
        )


def _stable_category_hash(value, seed):
    if isinstance(value, (bool, np.bool_)):
        payload = b"b:1" if value else b"b:0"
    elif isinstance(value, (int, np.integer)):
        payload = f"i:{int(value)}".encode("ascii")
    elif isinstance(value, str):
        payload = b"s:" + value.encode("utf-8")
    else:
        raise TypeError("Categorical hashing requires a string, boolean, or integer")
    result = 14695981039346656037 ^ (seed & _MASK)
    for byte in payload:
        result = ((result ^ byte) * 1099511628211) & _MASK
    return result


class CategoricalCodec:
    def __init__(self, mapping, dtype, cardinality=None):
        self.other = SPECIAL_TOKEN_IDS.other
        self.cardinality = bool(cardinality)
        self.missing = SPECIAL_TOKEN_IDS.unknown if cardinality else self.other
        self.hashing = (cardinality or {}).get("hashing")
        if self.hashing:
            buckets = {k: v for k, v in mapping.items() if is_hash_bucket_label(k)}
            self.bucket_count = int(self.hashing["num_buckets"])
            if not buckets or len(buckets) != self.bucket_count:
                raise ValueError(
                    "Categorical map does not contain the configured hash buckets"
                )
            self.bucket_start = min(buckets.values())
            self.buckets = buckets
            self.seed = int(self.hashing.get("seed", 0))
        mapping = {
            key: value
            for key, value in mapping.items()
            if not (self.hashing and is_hash_bucket_label(key))
            and not (cardinality and key == "[mask]")
        }
        if dtype == pl.Boolean or dtype.is_integer() or dtype.is_float():
            convert = (
                (lambda key: bool(int(key)))
                if dtype == pl.Boolean
                else (float if dtype.is_float() else int)
            )
            mapping = {
                convert(key): value
                for key, value in mapping.items()
                if key not in SPECIAL_TOKEN_LABELS
            }
        elif dtype in (pl.String, pl.Categorical, pl.Enum):
            mapping = {str(key): value for key, value in mapping.items()}
            if cardinality:
                mapping.update({"[unknown]": self.missing, "[other]": self.other})
        self.mapping = mapping

    def encode(self, value):
        if value is None:
            return self.missing
        if self.cardinality and value == "[mask]":
            raise ValueError("Reserved categorical value '[mask]'")
        if value in self.mapping:
            return self.mapping[value]
        if self.hashing:
            if value in self.buckets:
                return self.buckets[value]
            return (
                self.bucket_start
                + _stable_category_hash(value, self.seed) % self.bucket_count
            )
        return self.other

    def expression(self, name):
        source = pl.col(name)
        fallback = pl.lit(self.other, dtype=pl.Int64)
        if self.hashing:
            hashes = source.map_elements(
                lambda value: _stable_category_hash(value, self.seed),
                return_dtype=pl.UInt64,
            )
            fallback = (
                pl.lit(self.bucket_start, dtype=pl.UInt64)
                + hashes % pl.lit(self.bucket_count, dtype=pl.UInt64)
            ).cast(pl.Int64)
        if self.mapping:
            fallback = source.replace_strict(
                self.mapping, default=fallback, return_dtype=pl.Int64
            )
        return (
            pl.when(source.is_null()).then(self.missing).otherwise(fallback).alias(name)
        )
