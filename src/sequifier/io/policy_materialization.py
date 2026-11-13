"""Materialize policy histories using the persisted model execution schema."""

import torch

from sequifier.io.actions import decode_action
from sequifier.io.batch import SequifierInputs
from sequifier.model.execution_schema import ExecutionSchema


def encode_value(value, column, config, *, force_raw=False):
    if isinstance(value, (list, tuple)):
        return [
            encode_value(item, column, config, force_raw=force_raw) for item in value
        ]
    mapping = config.id_maps.get(column)
    if mapping and (force_raw or config.input_encoding == "raw"):
        if value in mapping:
            return mapping[value]
        if str(value) in mapping:
            return mapping[str(value)]
        from sequifier.special_tokens import SPECIAL_TOKEN_IDS

        special = {
            label: number for number, label in SPECIAL_TOKEN_IDS.labels_by_id.items()
        }
        if value in special and special[value] < config.categorical_n_classes[column]:
            return special[value]
        unknown = SPECIAL_TOKEN_IDS.unknown
        if unknown in mapping.values() or unknown < config.categorical_n_classes.get(
            column, 0
        ):
            return unknown
        raise ValueError(f"{column}: value is outside the persisted categorical codec")
    return value


def materialize_history(config, observations, previous_actions, static):
    schema = ExecutionSchema.from_dict(config.input_schema)
    rows = observations[-config.context_length :]
    previous = previous_actions[-config.context_length :]
    padding = config.context_length - len(rows)
    features, metadata = {}, {}
    for descriptor in schema.inputs:
        key = descriptor.key
        dtype = getattr(torch, descriptor.dtype)
        if key == "attention_valid_mask":
            metadata[key] = torch.tensor([[False] * padding + [True] * len(rows)])
            continue
        if key in config.static_columns:
            if key not in static:
                raise ValueError(f"missing static model input {key!r}")
            value = torch.as_tensor(
                encode_value(static[key], key, config), dtype=dtype
            ).unsqueeze(0)
        else:
            shape = (1, config.context_length, *descriptor.shape[2:])
            value = torch.zeros(shape, dtype=dtype)
            for index, row in enumerate(rows):
                if key in config.previous_action_inputs:
                    action = previous[index]
                    if action is None:
                        if key not in config.initial_inputs:
                            raise ValueError(
                                f"{key}: lagged action requires initial_inputs at episode start"
                            )
                        raw = config.initial_inputs[key]
                    else:
                        raw = decode_action(action, config)[
                            config.previous_action_inputs[key]
                        ]
                else:
                    if key not in row:
                        raise ValueError(f"missing temporal model input {key!r}")
                    raw = row[key]
                feedback = key in config.previous_action_inputs
                encoded = torch.as_tensor(
                    encode_value(raw, key, config, force_raw=feedback), dtype=dtype
                )
                if (
                    feedback
                    and config.input_encoding == "model"
                    and key in config.input_statistics
                ):
                    from sequifier.io.preprocessing import normalize_values

                    encoded = normalize_values(encoded, config.input_statistics[key])
                if tuple(encoded.shape) != tuple(shape[2:]):
                    raise ValueError(
                        f"{key}: payload shape differs from the execution schema"
                    )
                value[0, padding + index] = encoded
        (metadata if descriptor.role == "metadata" else features)[key] = value
    schema.validate(features, metadata)
    return SequifierInputs(features, metadata)
