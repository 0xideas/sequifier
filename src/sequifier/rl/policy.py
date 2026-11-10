"""Policy alignment adapter over Sequifier's configurable model system."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from sequifier.config.policy_config import ResolvedPolicyBinding
from sequifier.config.train_config import ResolvedSequifierConfig
from sequifier.io.batch import SequifierInputs
from sequifier.io.policy_window import normalize_policy_features
from sequifier.io.preprocessing import normalization_statistics
from sequifier.model.execution_schema import ExecutionSchema
from sequifier.model.prediction_distribution import PredictionScales
from sequifier.rl.distributions import CategoricalActionDistribution
from sequifier.rl.joint_distribution import JointActionDistribution

if TYPE_CHECKING:
    from sequifier.model.network import ComposableTransformerNetwork


def policy_execution_config(
    config: ResolvedPolicyBinding,
    model,
    device: str = "cpu",
    *,
    global_training=None,
    statistics=None,
    execution_model=None,
    **run_options,
):
    """Resolve policy alignment directly into the ordinary execution contract."""
    if execution_model is None:
        raise ValueError("joint policies require a resolved model execution contract")
    from sequifier.artifacts.model_config import resolved_config_from_model_config

    execution, _ = resolved_config_from_model_config(
        execution_model, device=device, interface_name=config.interface_name
    )
    from sequifier.rl.attachment import apply_binding

    apply_binding(execution, config.interface_name, config)
    for key, value in run_options.items():
        if key in type(execution).model_fields:
            setattr(execution, key, value)
    return execution


class Policy(nn.Module):
    """A convenience view; the network owns all trainable weights and execution."""

    config: ResolvedPolicyBinding
    execution_config: ResolvedSequifierConfig
    interface_name: str
    network: ComposableTransformerNetwork
    units: dict[str, str]
    ppo_policy_version: int
    grpo_policy_version: int

    @classmethod
    def from_network(cls, network, execution_config, interface_name: str):
        interface = execution_config.interface(interface_name)
        if interface.policy is None:
            raise ValueError(
                f"Model interface {interface_name!r} has no policy alignment"
            )
        policy = cls()
        policy.config = interface.policy
        policy.execution_config = execution_config
        policy.interface_name = interface_name
        policy.network = network
        policy._resolve_actions()
        return policy

    @property
    def interface(self):
        return self.execution_config.interface(self.interface_name)

    @property
    def device(self) -> torch.device:
        return next(self.network.parameters()).device

    @property
    def backbone(self):
        return self.network.backbone

    def _resolve_actions(self):
        from sequifier.model.decoders import AutoregressiveTransformerDecoderBranch

        route = self.network.resolve_interface(self.interface_name)
        prerequisites = {
            name: set(field.active_when) for name, field in self.config.actions.items()
        }
        for branch in route.decoder.branches.values():
            targets = [
                name for name in branch.target_columns if name in self.config.actions
            ]
            if isinstance(branch, AutoregressiveTransformerDecoderBranch) and targets:
                last = max(branch.target_columns.index(name) for name in targets)
                if set(branch.target_columns[: last + 1]) - set(targets):
                    raise ValueError(
                        "autoregressive actions must include all preceding decoder targets"
                    )
                for index, name in enumerate(targets):
                    prerequisites[name].update(targets[:index])
        pending = [
            name for name in self.interface.target_columns if name in prerequisites
        ]
        if set(pending) != set(self.config.actions):
            raise ValueError("policy actions must name existing model targets")
        order = []
        while pending:
            ready = next(
                (name for name in pending if prerequisites[name] <= set(order)), None
            )
            if ready is None:
                raise ValueError(
                    "action activity/decoder dependencies are cyclic or unknown"
                )
            order.append(ready)
            pending.remove(ready)
        self.action_order = tuple(order)
        self.autoregressive_actions = any(
            isinstance(branch, AutoregressiveTransformerDecoderBranch)
            and len(
                [name for name in branch.target_columns if name in self.config.actions]
            )
            > 1
            for branch in route.decoder.branches.values()
        )
        for index, name in enumerate(order):
            field = self.config.actions[name]
            if field.type != self.interface.target_column_types[name]:
                raise ValueError(f"{name}: action type differs from the model target")
            if set(field.active_when) - set(order[:index]):
                raise ValueError(
                    "activity conditions must reference earlier action fields"
                )
            if field.type == "categorical":
                self.action_decoder_ids(name)
                if field.active_when and field.inactive_value not in field.values:
                    raise ValueError(
                        f"{name}: inactive categorical value must be executable"
                    )
            for predecessor, labels in field.active_when.items():
                parent = self.config.actions[predecessor]
                if parent.type == "categorical" and any(
                    label not in parent.values for label in labels
                ):
                    raise ValueError(
                        f"{name}: activity condition uses an unknown category"
                    )
        if not hasattr(route, "prediction_scales"):
            route.prediction_scales = PredictionScales(self.config.actions).to(
                self.device
            )
        else:
            route.prediction_scales.ensure_fields(self.config.actions)
            route.prediction_scales.to(self.device)

    def action_decoder_ids(self, name):
        field = self.config.actions[name]
        mapping = self.interface.id_maps.get(name, {})
        ids = []
        for label in field.values:
            key = (
                label
                if isinstance(label, (str, int))
                else int(label)
                if label.is_integer()
                else str(label)
            )
            global_id = mapping.get(key, mapping.get(str(label), mapping.get(str(key))))
            if global_id is None:
                if (
                    isinstance(label, int)
                    and label in self.interface.target_decoder_ids[name]
                ):
                    global_id = label
                else:
                    raise ValueError(
                        f"{name}: action label {label!r} is absent from the model codec"
                    )
            if global_id not in self.interface.target_decoder_ids[name]:
                raise ValueError(f"{name}: action is absent from decoder support")
            ids.append(global_id)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(
                f"{name}: categorical actions need distinct executable decoder IDs"
            )
        return ids

    def set_normalization_statistics(
        self, statistics: Mapping[str, Mapping[str, float]]
    ) -> None:
        self.interface.selected_columns_statistics = (
            normalization_statistics(statistics)
            if self.interface.normalize_real_columns
            else {name: dict(values) for name, values in statistics.items()}
        )
        return

    def model_inputs(
        self, inputs: SequifierInputs
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Encode raw named histories without mutating stored data or masks."""
        self.validate_inputs(inputs)
        features = normalize_policy_features(
            inputs.inputs, inputs.metadata, self.interface
        )
        if any(
            t.is_floating_point() and not torch.isfinite(t).all()
            for t in features.values()
        ):
            raise ValueError("frozen input transforms produced non-finite values")
        return features, inputs.metadata

    def forward(self, inputs: SequifierInputs) -> Tensor | dict[str, Tensor]:
        return self.prepared_logits(*self.model_inputs(inputs))

    def prepared_logits(self, features, metadata) -> Tensor | dict[str, Tensor]:
        """Execute features already checked and normalized by model_inputs."""
        outputs = self.network(
            features, metadata, interface_name=self.interface_name
        ).logits
        return {
            name: value.masked_fill(
                ~metadata["attention_valid_mask"][:, -value.shape[1] :, None], 0.0
            )
            for name, value in outputs.items()
        }

    def distribution(
        self, inputs: SequifierInputs
    ) -> CategoricalActionDistribution | JointActionDistribution:
        return JointActionDistribution(self, inputs)

    def act(self, inputs: SequifierInputs, *, deterministic: bool, generator):
        distribution = self.distribution(inputs)
        action = distribution.select(deterministic, generator)
        return action, distribution.log_prob(action)

    def evaluate_actions(
        self, inputs: SequifierInputs, actions: Tensor | dict[str, Tensor]
    ) -> Tensor:
        return self.distribution(inputs).log_prob(actions)

    def validate_inputs(self, inputs: SequifierInputs) -> None:
        schema = ExecutionSchema.from_interface(
            self.interface, self.config.context_length
        )
        expected = {item.key for item in schema.inputs if item.role != "metadata"}
        if set(inputs.inputs) != expected:
            raise ValueError(
                "policy inputs must match the persisted named feature schema"
            )
        mask = inputs.attention_valid_mask
        if (
            mask.ndim != 2
            or mask.dtype != torch.bool
            or mask.shape[1] != self.config.context_length
            or mask.shape[0] < 1
        ):
            raise ValueError(
                "policy attention masks must be bool [batch, context_length]"
            )
        if not mask[:, -1].all() or (mask[:, :-1] & ~mask[:, 1:]).any():
            raise ValueError(
                "policy histories require left padding followed by valid positions"
            )
        ExecutionSchema.from_interface(
            self.interface, self.config.context_length
        ).validate(inputs.inputs, inputs.metadata)
        if any(
            value.device != self.device
            for value in [*inputs.inputs.values(), *inputs.metadata.values()]
        ):
            raise ValueError("policy features and masks must be on the model device")
        return
