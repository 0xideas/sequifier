"""Typed action conversion shared by datasets, collection, and algorithms."""

from collections.abc import Mapping

import torch

from sequifier.io.values import domain_scalar, numeric_scalar


def _metadata_key(name, config, prefix):
    key = prefix + name
    while key in config.actions:
        key = prefix + key
    return key


def action_latent_key(name, config):
    return _metadata_key(name, config, "__policy_latent:")


def action_component_key(name, config):
    return _metadata_key(name, config, "__policy_component:")


def action_record_fields(config):
    return set(config.actions) | {
        key(name, config)
        for name, field in config.actions.items()
        if field.type == "real"
        for key in (action_latent_key, action_component_key)
    }


def inactive_action(field):
    if field.type == "categorical":
        return field.values.index(field.inactive_value) if field.active_when else 0
    return field.inactive_value


def encode_action(value, config):
    if not isinstance(value, Mapping) or {
        key for key in value if key in config.actions
    } != set(config.actions):
        raise ValueError("actions must contain exactly the declared named fields")
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
    allowed = set(config.actions)
    for name, field in config.actions.items():
        if field.type != "real":
            continue
        key = action_latent_key(name, config)
        allowed.add(key)
        component_key = action_component_key(name, config)
        allowed.add(component_key)
        if component_key in value and value[component_key] is not None:
            component = numeric_scalar(value[component_key], component_key)
            if (
                component != int(component)
                or component < -2
                or component >= len(field.atoms)
            ):
                raise ValueError("invalid real mixture component")
            result[component_key] = int(component)
        if key in value and value[key] is not None:
            result[key] = numeric_scalar(value[key], key)
    if set(value) - allowed:
        raise ValueError("action record contains unknown fields")
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

    def tensor(key, dtype, *, default=None, required=False):
        return torch.tensor(
            [value[key] if required else value.get(key, default) for value in values],
            dtype=dtype,
            device=device,
        )

    result = {
        name: tensor(
            name,
            torch.long if field.type == "categorical" else real_dtype,
            required=True,
        )
        for name, field in config.actions.items()
    }
    latent_keys = {
        action_latent_key(name, config)
        for name, field in config.actions.items()
        if field.type == "real"
    }
    for key in latent_keys:
        result[key] = tensor(key, real_dtype, default=float("nan"))
    for name, field in config.actions.items():
        if field.type == "real":
            key = action_component_key(name, config)
            result[key] = tensor(key, torch.long, default=-2)
    return result


def action_record(actions, index=0):
    if isinstance(actions, dict):
        return {name: value[index].item() for name, value in actions.items()}
    return int(actions[index])


def decode_action(action, config):
    return {
        name: field.values[int(action[name])]
        if field.type == "categorical"
        else float(action[name])
        for name, field in config.actions.items()
    }
