"""Compile the policy binding against the model's decoder branches."""

from dataclasses import dataclass

import torch

from sequifier.config.policy_config import ActionFieldConfig
from sequifier.io.actions import (
    action_component_key,
    action_latent_key,
    inactive_action,
)
from sequifier.io.values import DomainValue
from sequifier.model.decoders import AutoregressiveTransformerDecoderBranch


@dataclass(frozen=True)
class ActionField:
    spec: ActionFieldConfig
    inactive: DomainValue
    activity: dict[str, tuple[DomainValue, ...]]
    support: tuple[int, ...]
    latent_key: str | None
    component_key: str | None


class ActionExecutionPlan:
    def __init__(self, actor, decoder):
        interface = actor.interface
        actions = actor.config.actions
        dependencies = {name: set(field.active_when) for name, field in actions.items()}
        self.independent = {}
        self.autoregressive = {}
        self.prefix_steps = {}
        for branch_name, branch in decoder.branches.items():
            targets = tuple(name for name in branch.target_columns if name in actions)
            if not targets:
                continue
            if isinstance(branch, AutoregressiveTransformerDecoderBranch):
                if tuple(branch.target_columns[: len(targets)]) != targets:
                    raise ValueError(
                        "autoregressive actions must include all preceding decoder targets"
                    )
                self.autoregressive[branch_name] = targets
                previous = None
                for name in targets:
                    self.prefix_steps[name] = (branch_name, previous)
                    if previous is not None:
                        dependencies[name].add(previous)
                    previous = name
            else:
                self.independent[branch_name] = targets

        pending = [name for name in interface.target_columns if name in actions]
        if set(pending) != set(actions):
            raise ValueError("policy actions must name existing model targets")
        order = []
        resolved = set()
        while pending:
            ready = next(
                (name for name in pending if dependencies[name] <= resolved), None
            )
            if ready is None:
                raise ValueError(
                    "action activity/decoder dependencies are cyclic or unknown"
                )
            order.append(ready)
            resolved.add(ready)
            pending.remove(ready)

        self.order = tuple(order)
        self.position = actor.config.decision_position
        self.has_autoregression = any(
            len(targets) > 1 for targets in self.autoregressive.values()
        )
        self.fields = {}
        record_fields = set(actions)
        for name in self.order:
            spec = actions[name]
            if spec.type != interface.target_column_types[name]:
                raise ValueError(f"{name}: action type differs from the model target")
            if spec.scale == "model" and name not in interface.prediction_distributions:
                raise ValueError(
                    f"{name}: scale: model requires a model prediction_distributions entry"
                )
            support = ()
            latent_key = component_key = None
            if spec.type == "categorical":
                decoder_indices = {
                    global_id: index
                    for index, global_id in enumerate(
                        interface.target_decoder_ids[name]
                    )
                }
                support = tuple(
                    decoder_indices[i] for i in actor.action_decoder_ids(name)
                )
                if spec.active_when and spec.inactive_value not in spec.values:
                    raise ValueError(
                        f"{name}: inactive categorical value must be executable"
                    )
            else:
                latent_key = action_latent_key(name, actor.config)
                component_key = action_component_key(name, actor.config)
                record_fields.update((latent_key, component_key))

            activity = {}
            for predecessor, labels in spec.active_when.items():
                parent = actions[predecessor]
                if parent.type == "categorical":
                    if any(label not in parent.values for label in labels):
                        raise ValueError(
                            f"{name}: activity condition uses an unknown category"
                        )
                    labels = [parent.values.index(label) for label in labels]
                activity[predecessor] = tuple(labels)
            self.fields[name] = ActionField(
                spec,
                inactive_action(spec),
                activity,
                support,
                latent_key,
                component_key,
            )
        self.record_fields = frozenset(record_fields)
        self._indices = {}

    def indices(self, device):
        # The shared network can move independently of its policy view.
        if device not in self._indices:
            # Replay autograd can reuse indices first materialized in inference.
            with torch.inference_mode(False):
                support = {
                    name: torch.tensor(field.support, dtype=torch.long, device=device)
                    for name, field in self.fields.items()
                    if field.support
                }
                activity = {
                    name: {
                        parent: torch.tensor(values, device=device)
                        for parent, values in field.activity.items()
                    }
                    for name, field in self.fields.items()
                    if field.activity
                }
            self._indices[device] = support, activity
        return self._indices[device]
