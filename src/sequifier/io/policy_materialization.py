"""Materialize policy histories with model input roles and fitted column dtypes."""

from typing import Any, cast

import numpy as np
import polars as pl
import torch

from sequifier.config.depth_layout import depth_mask_metadata_key
from sequifier.helpers import PANDAS_TO_TORCH_TYPES
from sequifier.io.actions import decode_action
from sequifier.io.batch import ModelEncodedPolicyInputs, SequifierInputs
from sequifier.io.categorical import CategoricalCodec, is_hash_bucket_label
from sequifier.io.preprocessing import normalize_input_values
from sequifier.special_tokens import SPECIAL_TOKEN_IDS

DESIRED_RETURN = "__policy_return"
_HISTORY_DTYPES = {
    "bool": torch.bool,
    "float32": torch.float32,
    "int64": torch.int64,
    "uint8": torch.uint8,
}


def encode_value(value, column, config, *, force_raw=False, codecs=None):
    mapping = config.id_maps.get(column)
    if not mapping or (config.input_encoding == "model" and not force_raw):
        return value
    if codecs is None:
        codecs = {}
    if isinstance(value, (list, tuple)):
        return [
            encode_value(item, column, config, force_raw=force_raw, codecs=codecs)
            for item in value
        ]
    storage = getattr(config, "raw_inputs", None)
    cardinality = storage.cardinality_config if storage else None
    if force_raw:
        if (
            not (cardinality or {}).get(column)
            and value is not None
            and str(value) in mapping
        ):
            return mapping[str(value)]
        if value in SPECIAL_TOKEN_IDS.ids_by_label:
            return SPECIAL_TOKEN_IDS.ids_by_label[value]
    if value is None or isinstance(value, str):
        dtype = pl.String
    elif isinstance(value, (bool, np.bool_)):
        dtype = pl.Boolean
    elif isinstance(value, (int, np.integer)):
        dtype = pl.Int64
    else:
        raise ValueError(f"{column}: expected a string, integer, or boolean category")
    key = (column, dtype)
    if key not in codecs:
        if cardinality is None and any(
            is_hash_bucket_label(label) for label in mapping
        ):
            raise ValueError(
                f"{column}: raw hashing requires the original cardinality settings"
            )
        codecs[key] = CategoricalCodec(mapping, dtype, (cardinality or {}).get(column))
    return codecs[key].encode(value)


def materialize_history(config, observations, previous_actions, static, *, codecs):
    schema = config.input_schema
    if schema is None:
        raise ValueError("Materializing policy windows requires a model input schema")
    transforms = config.transforms
    rows = observations[-config.context_length :]
    previous = previous_actions[-config.context_length :]
    if config.previous_action_inputs and len(previous) != len(rows):
        raise ValueError("lagged actions must align with the observation history")
    padding = config.context_length - len(rows)
    prepared = config.input_encoding == "model"
    normalize = prepared or bool(transforms.column_data_types)
    depth_masks = {
        column: depth_mask_metadata_key(name)
        for name, layout in schema.depth_layouts.items()
        for column in layout["columns"]
    }

    def tensor(raw, key, dtype, *, feedback=False):
        raw = encode_value(raw, key, config, force_raw=feedback, codecs=codecs)
        if dtype.is_floating_point and normalize and (not prepared or feedback):
            column_type = transforms.column_data_types.get(key, "")
            if transforms.normalize_real_columns and (
                statistics := transforms.selected_columns_statistics.get(key)
            ):
                raw = normalize_input_values(raw, statistics, column_type)
            elif column_type.lower() == "float16":
                raw = np.asarray(raw, dtype=np.float16)
        if dtype == torch.int64:
            values = torch.as_tensor(raw)
            if (
                values.is_floating_point()
                or values.is_complex()
                or values.dtype == torch.bool
            ):
                raise ValueError(f"{key}: expected integer categorical IDs")
            return values.to(dtype)
        return torch.as_tensor(raw, dtype=dtype)

    features, metadata = {}, {}
    categorical_storage = {}
    for descriptor in schema.inputs:
        key = descriptor.key
        dtype = _HISTORY_DTYPES[descriptor.dtype]
        if descriptor.role in {"feature", "static_feature"} and (
            column_type := transforms.column_data_types.get(key)
        ):
            storage_dtype = PANDAS_TO_TORCH_TYPES[column_type]
            if dtype.is_floating_point != storage_dtype.is_floating_point:
                raise ValueError(
                    f"{key}: {column_type} conflicts with its model input role"
                )
            if dtype == torch.int64 and storage_dtype != dtype:
                # Validate original IDs before a narrow cast can wrap them.
                categorical_storage[key] = storage_dtype
            else:
                dtype = storage_dtype
        if key == "attention_valid_mask":
            metadata[key] = torch.tensor([[False] * padding + [True] * len(rows)])
            continue
        if key in config.static_columns:
            if key not in static:
                raise ValueError(f"missing static model input {key!r}")
            payload = tensor(static[key], key, dtype).unsqueeze(0)
        else:
            shape = (1, config.context_length, *descriptor.shape[2:])
            payload = torch.zeros(shape, dtype=dtype)
            for index, row in enumerate(rows):
                raw: Any
                feedback = key in config.previous_action_inputs
                if feedback:
                    if previous[index] is not None:
                        raw = decode_action(previous[index], config)[
                            config.previous_action_inputs[key]
                        ]
                    elif key in config.initial_inputs:
                        raw = config.initial_inputs[key]
                    else:
                        raise ValueError(
                            f"{key}: lagged action requires a feedback initial value at episode start"
                        )
                elif key in row:
                    raw = row[key]
                else:
                    raise ValueError(f"missing temporal model input {key!r}")

                if key == DESIRED_RETURN and config.return_conditioned and normalize:
                    raw /= config.returns.scale
                    feedback = True
                mask = row.get(config.raw_mask_column, False) if not prepared else False
                if mask not in (True, False):
                    raise ValueError(
                        f"{config.raw_mask_column}: expected a boolean or 0 or 1"
                    )
                if mask and key in transforms.column_data_types and not feedback:
                    value = torch.tensor(
                        SPECIAL_TOKEN_IDS.mask if dtype == torch.int64 else 0,
                        dtype=dtype,
                    )
                elif key in depth_masks and not prepared:
                    valid = torch.as_tensor(row[depth_masks[key]])
                    depth_values = cast(Any, raw)
                    if valid.dtype != torch.bool or len(valid) != len(depth_values):
                        raise ValueError(
                            f"{key}: depth mask must be boolean and match the values"
                        )
                    value = torch.zeros(len(valid), dtype=dtype)
                    if valid.any():
                        value[valid] = tensor(
                            [
                                item
                                for item, keep in zip(depth_values, valid.tolist())
                                if keep
                            ],
                            key,
                            dtype,
                        )
                else:
                    value = tensor(raw, key, dtype, feedback=feedback)
                if tuple(value.shape) != tuple(shape[2:]):
                    raise ValueError(
                        f"{key}: payload shape differs from the execution schema"
                    )
                payload[0, padding + index] = value
        (metadata if descriptor.role == "metadata" else features)[key] = payload
    schema.validate(features, metadata)
    for key, storage_dtype in categorical_storage.items():
        values = features[key]
        max_id = torch.iinfo(storage_dtype).max
        classes = schema.n_classes.get(key)
        # Schema validation already checked [0, classes) for the entire window.
        if classes is None or classes - 1 > max_id:
            upper = min(max_id, torch.iinfo(torch.int64).max)
            if (values < 0).any().item() or (values > upper).any().item():
                raise ValueError(
                    f"{key}: categorical ID outside the range of {storage_dtype}"
                )
        features[key] = values.to(storage_dtype)
    input_type = ModelEncodedPolicyInputs if normalize else SequifierInputs
    return input_type(features, metadata)
