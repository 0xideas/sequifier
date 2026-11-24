"""Arrow column types shared by imported episodes and collected decisions."""

import sys

import pyarrow as pa

from sequifier.io.action_record import action_record_layout
from sequifier.io.sequence import CONTEXT_COLUMN
from sequifier.trajectory.tables import (
    ACTION_RECORD,
    GROUP_COLUMN,
    PREVIOUS_ACTION_RECORD,
    REWARD_COLUMN,
    SEQUENCE_NAME_COLUMN,
    SUCCESSOR_PREFIX,
    TRANSITION_FLAGS,
)

MAX_FLOAT64_INTEGER = int(sys.float_info.max)


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _float64_value(value, dtype, column):
    if value is None:
        return value
    if pa.types.is_float64(dtype) and _is_integer(value):
        if abs(value) > MAX_FLOAT64_INTEGER:
            raise ValueError(f"{column}: integer exceeds the finite float64 range")
        converted = float(value)
        if converted != value:
            raise ValueError(
                f"{column}: integer cannot be represented exactly as float64"
            )
        return converted
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return [_float64_value(item, dtype.value_type, column) for item in value]
    if pa.types.is_struct(dtype):
        return {
            name: _float64_value(item, dtype.field(name).type, f"{column}.{name}")
            for name, item in value.items()
        }
    return value


def _checked_array(entries, dtype, column):
    dtype = dtype if dtype is not None else pa.infer_type(entries)
    return pa.array(
        [_float64_value(value, dtype, column) for value in entries], type=dtype
    )


def cast_episode_table(table, schema):
    if table.schema == schema:
        return table
    arrays = []
    for field in schema:
        if field.name == CONTEXT_COLUMN and field.name not in table.column_names:
            arrays.append(pa.nulls(len(table), type=field.type))
            continue
        column = table[field.name]
        arrays.append(
            column
            if column.type == field.type
            else _checked_array(column.to_pylist(), field.type, field.name)
        )
    return pa.Table.from_arrays(arrays, schema=schema)


def _action_type(field):
    if field.type == "real":
        return pa.float64()
    if not field.values:
        raise ValueError("categorical sequence actions require declared values")
    if all(isinstance(label, str) for label in field.values):
        return pa.string()
    if any(isinstance(label, str) for label in field.values):
        raise ValueError(
            "categorical sequence actions cannot mix string and numeric labels"
        )
    if all(_is_integer(label) for label in field.values):
        if min(field.values) >= 0 and max(field.values) >= 2**63:
            return pa.uint64()
        return pa.int64()
    return pa.float64()


def _action_record_type(policy):
    auxiliary_types = {"latent": pa.float64(), "component": pa.int64()}
    return pa.struct(
        [
            pa.field(
                field.key,
                auxiliary_types[field.kind]
                if field.kind in auxiliary_types
                else _action_type(policy.actions[field.action]),
            )
            for field in action_record_layout(policy).fields
        ]
    )


def _policy_feature_types(policy):
    types = {}
    if policy is None or policy.input_schema is None:
        return types
    for field in policy.input_schema.inputs:
        if field.dtype == "int64":
            # The model accepts token IDs, but sequence files store domain labels.
            continue
        dtype = {"float32": pa.float64(), "bool": pa.bool_(), "uint8": pa.uint8()}[
            field.dtype
        ]
        axes = 1 if field.key in policy.static_columns else 2
        for _ in field.shape[axes:]:
            dtype = pa.list_(dtype)
        types[field.key] = dtype
    return types


def sequence_table(values, observations, *, policy=None, collection_type=None):
    """Encode nullable transition columns with the types of their domain fields."""
    length = len(values["sequenceId"])
    values = {**values, CONTEXT_COLUMN: values.get(CONTEXT_COLUMN, [None] * length)}
    types = {
        **_policy_feature_types(policy),
        "sequenceId": pa.int64(),
        "itemPosition": pa.int64(),
        SEQUENCE_NAME_COLUMN: pa.string(),
        CONTEXT_COLUMN: pa.string(),
        GROUP_COLUMN: pa.string(),
        REWARD_COLUMN: pa.float64(),
        **{name: pa.bool_() for name in TRANSITION_FLAGS},
    }
    if ACTION_RECORD in values:
        action_type = (
            _action_record_type(policy)
            if policy is not None
            else pa.infer_type(values[ACTION_RECORD])
        )
        types[ACTION_RECORD] = types[PREVIOUS_ACTION_RECORD] = action_type
        for field in action_type:
            types["__prediction_" + field.name] = field.type
    for name in observations:
        successor = SUCCESSOR_PREFIX + name
        dtype = types.get(name)
        if dtype is None:
            dtype = pa.infer_type([*values[name], *values[successor]])
        types[name] = types[successor] = dtype
    if collection_type is not None:
        types["__collection"] = collection_type
    return pa.table(
        {
            name: _checked_array(entries, types.get(name), name)
            for name, entries in values.items()
        }
    )
