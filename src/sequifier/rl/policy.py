"""Policy alignment adapter over Sequifier's configurable model system."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from sequifier.config.policy_config import ResolvedPolicyBinding
from sequifier.io.batch import SequifierInputs
from sequifier.io.policy_window import checked_policy_features
from sequifier.io.preprocessing import normalization_statistics
from sequifier.model.execution import ModelExecution
from sequifier.model.prediction_distribution import PredictionScales
from sequifier.rl.action_plan import ActionDecision, ActionExecutionPlan
from sequifier.rl.joint_distribution import JointActionDistribution

if TYPE_CHECKING:
    from sequifier.model.network import ComposableTransformerNetwork


class ExecutablePolicy(nn.Module):
    """Shared input contract for native and exported policy execution."""

    execution_config: ModelExecution
    interface_name: str
    config: ResolvedPolicyBinding

    @property
    def interface(self):
        return self.execution_config.interface(self.interface_name)

    @property
    def units(self) -> dict[str, str]:
        return self.config.units

    def model_inputs(
        self, inputs: SequifierInputs, *, input_encoding=None
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Encode raw named histories without mutating stored data or masks."""
        self.validate_inputs(inputs)
        features = checked_policy_features(
            inputs, self.interface, input_encoding=input_encoding
        )
        return features, inputs.metadata

    def validate_inputs(self, inputs: SequifierInputs) -> None:
        schema = self.config.input_schema
        assert schema is not None
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
        schema.validate(inputs.inputs, inputs.metadata)
        if any(
            value.device != self.device
            for value in [*inputs.inputs.values(), *inputs.metadata.values()]
        ):
            raise ValueError("policy features and masks must be on the model device")
        return


class Policy(ExecutablePolicy):
    """A convenience view; the network owns all trainable weights and execution."""

    config: ResolvedPolicyBinding
    execution_config: ModelExecution
    interface_name: str
    network: ComposableTransformerNetwork
    transfer_report: dict | None
    ppo_policy_version: int = 0
    grpo_policy_version: int = 0

    def __init__(self):
        super().__init__()
        self.transfer_report = None
        self.role_call = None

    @classmethod
    def from_network(cls, network, execution_config, binding: ResolvedPolicyBinding):
        policy = cls()
        policy.config = binding
        policy.execution_config = execution_config
        policy.interface_name = binding.interface_name
        policy.network = network
        policy._resolve_actions()
        return policy

    @property
    def device(self) -> torch.device:
        if self.role_call is not None:
            return self.role_call.device
        return next(self.network.parameters()).device

    @property
    def backbone(self):
        return self.network.backbone

    def _resolve_actions(self):
        route = self.network.resolve_interface(self.interface_name)
        self.action_plan = ActionExecutionPlan(self, route.require_decoder())
        self.action_order = self.action_plan.order
        self.autoregressive_actions = self.action_plan.has_autoregression
        if route.prediction_scales is None:
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
        self, inputs: SequifierInputs, *, input_encoding=None
    ) -> JointActionDistribution:
        return JointActionDistribution(self, inputs, input_encoding=input_encoding)

    def predict_actions(self, inputs, deterministic, generator, *, probabilities):
        distribution = self.distribution(inputs, input_encoding="model")
        if probabilities:
            selected = distribution.select_and_score(
                deterministic, generator, include_probabilities=True
            )
            return (
                distribution.domain_actions(selected.decision),
                selected.log_prob,
                selected.probabilities,
            )
        decision = distribution.select_decision(deterministic, generator)
        return distribution.domain_actions(decision), None, {}

    def action_scores(
        self,
        inputs,
        actions=None,
        *,
        deterministic=False,
        generator=None,
        entropy=False,
    ):
        if self.role_call is not None:
            return self.role_call(
                inputs,
                actions,
                deterministic=deterministic,
                generator=generator,
                entropy=entropy,
            )
        return self.score_distribution(
            inputs,
            actions,
            deterministic=deterministic,
            generator=generator,
            entropy=entropy,
        )

    def score_distribution(
        self,
        inputs,
        actions=None,
        *,
        deterministic=False,
        generator=None,
        entropy=False,
    ):
        distribution = self.distribution(inputs)
        if actions is None:
            selection = distribution.select_and_score(deterministic, generator)
            record, score = selection.actions, selection.log_prob
        else:
            record, score = {}, distribution.log_prob(actions)
        return record, score, distribution.entropy(generator) if entropy else None

    def act(self, inputs: SequifierInputs, *, deterministic: bool, generator):
        record, score, _ = self.action_scores(
            inputs, deterministic=deterministic, generator=generator
        )
        return record, score

    def evaluate_actions(
        self,
        inputs: SequifierInputs,
        actions: ActionDecision | Mapping[str, Tensor],
    ) -> Tensor:
        return self.action_scores(inputs, actions)[1]
