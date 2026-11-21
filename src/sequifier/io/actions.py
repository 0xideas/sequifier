"""Typed action conversion shared by datasets, collection, and algorithms."""

from collections.abc import Mapping

import torch

from sequifier.io.action_record import action_record_layout
from sequifier.io.values import domain_scalar, numeric_scalar


def inactive_action(field):
    if field.type == "categorical":
        return field.values.index(field.inactive_value) if field.active_when else 0
    return field.inactive_value


def encode_action(value, config):
    if not isinstance(value, Mapping):
        raise ValueError("actions must contain exactly the declared named fields")
    if missing := config.actions.keys() - value.keys():
        raise ValueError(f"action record lacks fields: {sorted(missing)}")
    result = {}
    for name, field in config.actions.items():
        item = domain_scalar(value[name], name)
        if field.type == "categorical":
            if item not in field.values:
                raise ValueError(f"{name}: action is outside the executable vocabulary")
            item = field.values.index(item)
        else:
            item = numeric_scalar(item, name)
        result[name] = item
    layout = action_record_layout(config)
    if value.keys() - layout.keys:
        raise ValueError("action record contains unknown fields")
    latents, components = layout.unpack_auxiliary(value)
    for name, item in components.items():
        if item is None:
            continue
        key = layout.components[name]
        component = numeric_scalar(item, key)
        if component != int(component) or not -2 <= component < len(
            config.actions[name].atoms
        ):
            raise ValueError("invalid real mixture component")
        result[key] = int(component)
    for name, item in latents.items():
        if item is not None:
            key = layout.latents[name]
            result[key] = numeric_scalar(item, key)
    return result


def action_batch(values, config, *, device=None, encoded=True):
    values = list(values)
    if not encoded:
        values = [encode_action(value, config) for value in values]
    real_dtype = (
        torch.float32
        if device is not None and torch.device(device).type == "mps"
        else torch.float64
    )

    result = {}
    for field in action_record_layout(config).fields:
        missing = field.missing
        entries = (
            [value[field.key] for value in values]
            if missing is None
            else [value.get(field.key, missing) for value in values]
        )
        dtype = torch.long if field.kind in {"categorical", "component"} else real_dtype
        result[field.key] = torch.tensor(entries, dtype=dtype, device=device)
    return result


def action_record(actions, index=0):
    if isinstance(actions, dict):
        return {name: value[index].item() for name, value in actions.items()}
    return int(actions[index])


def decode_action(action, config):
    return {
        name: decode_action_value(action[name], field)
        for name, field in config.actions.items()
    }


def decode_action_value(value, field):
    return field.values[int(value)] if field.type == "categorical" else float(value)
