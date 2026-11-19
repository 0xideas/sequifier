"""Compile the policy binding against the model's decoder branches."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor

from sequifier.config.policy_config import ActionFieldConfig
from sequifier.io.actions import (
    action_component_key,
    action_latent_key,
    decode_action_value,
    inactive_action,
)
from sequifier.io.preprocessing import normalize_values
from sequifier.io.values import DomainValue
from sequifier.model.decoders import AutoregressiveTransformerDecoderBranch


@dataclass
class ActionDecision:
    """Batched executable values and the real-valued data needed to replay them."""

    values: dict[str, Tensor] = field(default_factory=dict)
    latents: dict[str, Tensor] = field(default_factory=dict)
    components: dict[str, Tensor] = field(default_factory=dict)

    def cpu(self) -> "ActionDecision":
        return ActionDecision(
            {name: value.cpu() for name, value in self.values.items()},
            {name: value.cpu() for name, value in self.latents.items()},
            {name: value.cpu() for name, value in self.components.items()},
        )


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
        self.prefix_steps = {}
        self.has_autoregression = False
        for branch_name, branch in decoder.branches.items():
            targets = tuple(name for name in branch.target_columns if name in actions)
            if not targets:
                continue
            if isinstance(branch, AutoregressiveTransformerDecoderBranch):
                if tuple(branch.target_columns[: len(targets)]) != targets:
                    raise ValueError(
                        "autoregressive actions must include all preceding decoder targets"
                    )
                self.has_autoregression |= len(targets) > 1
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
        self.action_names = tuple(actions)
        self._indices = {}

    def from_record(self, actions: Mapping[str, Tensor]) -> ActionDecision:
        """Translate a table or trajectory batch into the internal decision."""
        if actions.keys() - self.record_fields:
            raise ValueError("joint action contains unknown fields")
        return ActionDecision(
            {name: actions[name] for name in self.action_names if name in actions},
            {
                name: actions[field.latent_key]
                for name, field in self.fields.items()
                if field.latent_key is not None and field.latent_key in actions
            },
            {
                name: actions[field.component_key]
                for name, field in self.fields.items()
                if field.component_key is not None and field.component_key in actions
            },
        )

    def to_record(self, decision: ActionDecision) -> dict[str, Tensor]:
        """Pack a decision in the existing flat action record format."""
        actions = {name: decision.values[name] for name in self.order}
        for name in self.order:
            field = self.fields[name]
            if field.latent_key is not None and name in decision.latents:
                actions[field.latent_key] = decision.latents[name]
            if field.component_key is not None and name in decision.components:
                actions[field.component_key] = decision.components[name]
        return actions

    def domain_action(
        self, decision: ActionDecision, index: int = 0
    ) -> dict[str, DomainValue]:
        return {
            name: decode_action_value(
                decision.values[name][index].item(), self.fields[name].spec
            )
            for name in self.action_names
        }

    def outputs(
        self, decoder, context, valid, statistics, decision: ActionDecision
    ) -> Iterator[tuple[str, Tensor]]:
        """Yield each decoder output using the values chosen at earlier steps."""
        support, _ = self.indices(context.device)
        outputs = {}
        for branch_name, targets in self.independent.items():
            branch = decoder.branches[branch_name]
            outputs.update(branch.decode_targets(context, targets))
        prefixes = {}
        for name in self.order:
            if name not in self.prefix_steps:
                yield name, outputs[name]
                continue
            branch_name, previous = self.prefix_steps[name]
            branch = decoder.branches[branch_name]
            if previous is None:
                prefix = branch.start_prefix(context)
            else:
                teacher = self._teacher(
                    decision.values.get(previous),
                    support.get(previous),
                    valid,
                    statistics.get(previous),
                )
                prefix = branch.append_prefix(prefixes[branch_name], previous, teacher)
            prefixes[branch_name] = prefix
            yield name, branch.decode_prefix(prefix, (name,))[name]

    @staticmethod
    def _teacher(value, support, valid, stats):
        if value is None:
            dtype = torch.long if support is not None else torch.float32
            return valid.new_zeros(valid.shape, dtype=dtype)
        if support is not None:
            return support[value.long()].masked_fill(~valid, 0)
        teacher = normalize_values(value, stats) if stats is not None else value
        return teacher.float().expand(valid.shape[0])

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
