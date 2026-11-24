"""Reserved names in authored feature and output column declarations."""

PREDICTION_COORDINATES = (
    "sequenceId",
    "subsequenceId",
    "windowStartOffset",
    "itemPosition",
)
ACTION_LOG_PROBABILITY = "action_log_probability"
TRANSITION_COLUMNS = ("reward", "terminated", "truncated", "bootstrap_allowed")
_STORAGE_COLUMNS = {
    "sequenceName",
    "startItemPosition",
    "leftPadLength",
    "rightPadLength",
    "splitStartItemPosition",
    "splitEndItemPosition",
    "targetValidFrom",
    "inputCol",
    "attention_valid_mask",
    "target_valid_mask",
}


def validate_column_names(names, field, *, allowed=(), prefixes=()):
    """Reject storage and output names while allowing explicit recorded-field references."""
    reserved = set(PREDICTION_COORDINATES) | _STORAGE_COLUMNS | {ACTION_LOG_PROBABILITY}
    conflicts = sorted(
        name
        for name in names
        if (name in reserved or name.startswith(("__", "depth_valid_mask:")))
        and name not in allowed
        and (name == "__executed_action" or not name.startswith(prefixes))
    )
    if conflicts:
        raise ValueError(f"{field} uses reserved Sequifier column names: {conflicts}")


def validate_policy_column_types(types, *, temporal_columns=()):
    """Canonicalize authored policy storage dtypes without silently narrowing them."""
    from sequifier.helpers import canonicalize_polars_dtype_name, is_integer_dtype_name

    normalized = {}
    for name, dtype in (types or {}).items():
        if name in temporal_columns and dtype.startswith(("Date", "Datetime")):
            normalized[name] = dtype
            continue
        try:
            spelling = dtype.strip().lower()
            if spelling.startswith("uint"):
                spelling = "UInt" + spelling[4:]
            else:
                spelling = spelling.capitalize()
            canonical = canonicalize_polars_dtype_name(spelling)
        except ValueError:
            canonical = None
        if canonical is None or (
            canonical != "Float32" and not is_integer_dtype_name(canonical)
        ):
            raise ValueError(
                f"column_data_types.{name}: policy real columns require Float32 "
                f"and categorical columns require an integer dtype, got {dtype!r}"
            )
        normalized[name] = canonical
    return normalized if types is not None else None
