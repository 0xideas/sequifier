"""Policy alignment adapter over Sequifier's configurable model system."""

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from sequifier.artifacts.model_config import resolved_config_from_model_config
from sequifier.config.policy_config import CategoricalPolicyConfig
from sequifier.io.batch import SequifierInputs
from sequifier.io.policy_window import (
    DESIRED_RETURN,
    PREVIOUS_ACTION,
    action_global_ids,
    action_target,
)
from sequifier.model.execution_schema import ExecutionSchema
from sequifier.model.factory import build_transformer_network
from sequifier.model.ingestions import get_feature_embedding_dims
from sequifier.rl.distributions import CategoricalActionDistribution


def policy_execution_config(config: CategoricalPolicyConfig, device: str = "cpu"):
    """Resolve policy features through the same execution schema as forecasts.

    Actions are decoder IDs; the extra input-only ID is the episode-start token.
    Returns and previous actions remain distinct from the current action target.
    """
    columns = config.observation_columns + config.static_columns + [PREVIOUS_ACTION]
    if config.return_conditioned:
        columns.append(DESIRED_RETURN)
    target = action_target(config)
    categorical = list(config.categorical_n_classes) + [PREVIOUS_ACTION, target]
    real = [column for column in columns if column not in categorical]
    decoder_ids = action_global_ids(config)
    action_width = decoder_ids[-1] + 1
    lookup = [-1] * action_width
    for local_id, global_id in enumerate(decoder_ids):
        lookup[global_id] = local_id
    n_classes = {
        **config.categorical_n_classes,
        PREVIOUS_ACTION: action_width + 1,
        target: action_width,
    }
    categorical_inputs = [column for column in columns if column in categorical]
    width = max(config.backbone.dim_model, len(real) + 2 * len(categorical_inputs))
    default_ingestion = {
        "type": "embedding",
        "output_dim": width,
        "feature_embedding_dims": {
            **{column: 1 for column in real},
            **get_feature_embedding_dims(
                width - len(real), categorical_inputs, [], n_classes
            ),
        },
    }
    values = {
        "training_objective": "policy",
        "context_length": config.context_length,
        "target_offset": 0,
        "backbone": (
            config.backbone_component.model_dump(mode="json")
            if config.backbone_component is not None
            else {"architecture": config.backbone.model_dump(mode="json")}
        ),
        "interfaces": {
            config.interface_name: {
                "policy": config.model_dump(mode="json"),
                "input_columns": columns,
                "target_columns": [target],
                "column_data_types": {
                    column: "Int64" if column in categorical else "Float32"
                    for column in columns + [target]
                },
                "categorical_columns": categorical,
                "real_columns": real,
                "target_column_types": {target: "categorical"},
                "n_classes": n_classes,
                "target_decoder_ids": {target: decoder_ids},
                "target_global_to_decoder": {target: lookup},
                "id_maps": {
                    **config.id_maps,
                    target: {
                        str(value): global_id
                        for value, global_id in zip(config.action_values, decoder_ids)
                    },
                },
                "normalize_real_columns": True,
                "selected_columns_statistics": {
                    column: {"mean": 0.0, "std": 1.0}
                    for column in real
                    if column != DESIRED_RETURN
                },
                "ingestion": (
                    config.ingestion.model_dump(mode="json")
                    if config.ingestion
                    else default_ingestion
                ),
                "decoder": (
                    config.decoder.model_dump(mode="json")
                    if config.decoder
                    else {
                        "type": "linear",
                        "prediction_length": 1,
                        "support": 1,
                    }
                ),
            }
        },
    }
    return resolved_config_from_model_config(values, device=device)[0]


class CategoricalPolicy(nn.Module):
    """A convenience view; the network owns all trainable weights and execution."""

    def __init__(self, config: CategoricalPolicyConfig, *, device: str = "cpu"):
        super().__init__()
        self.config = config
        self.execution_config = policy_execution_config(config, device)
        self.network = build_transformer_network(
            self.execution_config, device=torch.device(device)
        ).network
        self.interface_name = config.interface_name

    @classmethod
    def from_network(cls, network, execution_config, interface_name: str):
        interface = next(
            dataset.interface
            for dataset in execution_config.dataset_training.values()
            if dataset.model_interface == interface_name
        )
        if interface.policy is None:
            raise ValueError(
                f"Model interface {interface_name!r} has no policy alignment"
            )
        policy = cls.__new__(cls)
        nn.Module.__init__(policy)
        policy.config = interface.policy
        policy.execution_config = execution_config
        policy.interface_name = interface_name
        policy.network = network
        return policy

    @property
    def interface(self):
        return next(
            dataset.interface
            for dataset in self.execution_config.dataset_training.values()
            if dataset.model_interface == self.interface_name
        )

    @property
    def device(self) -> torch.device:
        return next(self.network.parameters()).device

    @property
    def backbone(self):
        return self.network.backbone

    def set_normalization_statistics(
        self, statistics: Mapping[str, Mapping[str, float]]
    ) -> None:
        expected = (
            set(self.config.observation_columns + self.config.static_columns)
            - self.config.categorical_n_classes.keys()
        )
        if set(statistics) != expected:
            raise ValueError("normalization statistics must match real policy columns")
        fitted = {}
        for column, values in statistics.items():
            mean, std = float(values["mean"]), float(values["std"])
            if not math.isfinite(mean) or not math.isfinite(std) or std < 0:
                raise ValueError(
                    "policy normalization requires finite means and nonnegative scales"
                )
            fitted[column] = {"mean": mean, "std": std}
        self.interface.selected_columns_statistics = fitted

    def model_inputs(
        self, inputs: SequifierInputs
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Encode raw named histories without mutating stored data or masks."""
        self.validate_inputs(inputs)
        features = dict(inputs.inputs)
        mask = inputs.attention_valid_mask
        for column, statistics in self.interface.selected_columns_statistics.items():
            features[column] = (
                (features[column] - statistics["mean"]) / (statistics["std"] + 1e-9)
            ).masked_fill(~mask, 0.0)
        if self.config.return_conditioned:
            features[DESIRED_RETURN] = (
                features[DESIRED_RETURN] / self.config.returns.scale
            )
        return features, inputs.metadata

    def forward(self, inputs: SequifierInputs) -> Tensor:
        features, metadata = self.model_inputs(inputs)
        logits = self.network(
            features, metadata, interface_name=self.interface_name
        ).logits[self.interface.target_columns[0]]
        return logits.masked_fill(
            ~inputs.attention_valid_mask[:, -logits.shape[1] :, None], 0.0
        )

    def distribution(self, inputs: SequifierInputs) -> CategoricalActionDistribution:
        return CategoricalActionDistribution(self(inputs))

    def validate_inputs(self, inputs: SequifierInputs) -> None:
        expected = set(self.interface.input_columns)
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
        ids = action_global_ids(self.config)
        allowed = torch.tensor(ids + [ids[-1] + 1], device=self.device)
        if not torch.isin(inputs.inputs[PREVIOUS_ACTION], allowed).all():
            raise ValueError(
                "previous actions must be executable action IDs or the input-only start token"
            )
