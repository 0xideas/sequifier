"""Policy alignment adapter over Sequifier's configurable model system."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from sequifier.config.policy_config import ResolvedPolicyBinding
from sequifier.io.batch import SequifierInputs
from sequifier.io.policy_window import normalize_policy_features
from sequifier.io.preprocessing import normalization_statistics
from sequifier.model.execution import ModelExecution
from sequifier.model.execution_schema import ExecutionSchema
from sequifier.model.prediction_distribution import PredictionScales
from sequifier.rl.action_plan import ActionExecutionPlan
from sequifier.rl.joint_distribution import JointActionDistribution

if TYPE_CHECKING:
    from sequifier.model.network import ComposableTransformerNetwork


class Policy(nn.Module):
    """A convenience view; the network owns all trainable weights and execution."""

    config: ResolvedPolicyBinding
    execution_config: ModelExecution
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
        policy.units = dict(interface.policy.units)
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
        route = self.network.resolve_interface(self.interface_name)
        self.action_plan = ActionExecutionPlan(self, route.decoder)
        self.action_order = self.action_plan.order
        self.autoregressive_actions = self.action_plan.has_autoregression
        if not hasattr(route, "prediction_scales"):
            route.prediction_scales = PredictionScales(
                self.interface.prediction_distributions
            ).to(self.device)
        else:
            route.prediction_scales.ensure_fields(
                self.interface.prediction_distributions
            )
            route.prediction_scales.to(self.device)

    def action_decoder_ids(self, name):
        field = self.config.actions[name]
        mapping = self.interface.id_maps.get(name, {})
        support = set(self.interface.target_decoder_ids.get(name, ()))
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
                if isinstance(label, int) and label in support:
                    global_id = label
                else:
                    raise ValueError(
                        f"{name}: action label {label!r} is absent from the model codec"
                    )
            if global_id not in support:
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

    def distribution(self, inputs: SequifierInputs) -> JointActionDistribution:
        return JointActionDistribution(self, inputs)

    def act(self, inputs: SequifierInputs, *, deterministic: bool, generator):
        selected = self.distribution(inputs).select_and_score(deterministic, generator)
        return selected.actions, selected.log_prob

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
