"""Policy alignment adapter over Sequifier's configurable model system."""

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from sequifier.artifacts.model_config import resolved_config_from_model_config
from sequifier.config.policy_config import CategoricalPolicyConfig
from sequifier.model.factory import build_transformer_network
from sequifier.model.ingestions import get_feature_embedding_dims
from sequifier.rl.distributions import CategoricalActionDistribution
from sequifier.rl.history import PolicyInputs

PREVIOUS_ACTION = "__policy_previous_action"
DESIRED_RETURN = "__policy_return"


def policy_execution_config(config: CategoricalPolicyConfig, device: str = "cpu"):
    """Resolve policy features through the same execution schema as forecasts.

    Actions are decoder IDs; the extra input-only ID is the episode-start token.
    Returns and previous actions remain distinct from the current action target.
    """
    columns = config.observation_columns + config.static_columns + [PREVIOUS_ACTION]
    if config.return_conditioned:
        columns.append(DESIRED_RETURN)
    target = (
        "__policy_action" if config.action_column in columns else config.action_column
    )
    categorical = list(config.observation_n_classes) + [PREVIOUS_ACTION, target]
    real = [column for column in columns if column not in categorical]
    count = len(config.action_values)
    n_classes = {
        **config.observation_n_classes,
        PREVIOUS_ACTION: count + 1,
        target: count,
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
        "training_objective": "causal",
        "context_length": config.context_length,
        "target_offset": 1,
        "backbone": {"architecture": config.backbone.model_dump(mode="json")},
        "interfaces": {
            "policy": {
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
                "target_decoder_ids": {target: list(range(count))},
                "target_global_to_decoder": {target: list(range(count))},
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
        self.interface_name = "policy"

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
            - self.config.observation_n_classes.keys()
        )
        if set(statistics) != expected:
            raise ValueError("normalization statistics must match real policy columns")
        fitted = {}
        for column, values in statistics.items():
            mean, std = float(values["mean"]), float(values["std"])
            if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
                raise ValueError(
                    "policy normalization requires finite means and positive scales"
                )
            fitted[column] = {"mean": mean, "std": std}
        self.interface.selected_columns_statistics = fitted

    def set_normalization(self, observations: Tensor, static_inputs: Tensor) -> None:
        """Convenience API for in-memory callers; file training fits incrementally."""
        statistics = {}
        for values, columns in (
            (observations, self.config.observation_columns),
            (static_inputs, self.config.static_columns),
        ):
            if (
                values.ndim != 2
                or values.shape[1] != len(columns)
                or not len(values)
                or not torch.isfinite(values).all()
            ):
                raise ValueError(
                    "normalization requires finite rows matching policy columns"
                )
            for index, column in enumerate(columns):
                if column not in self.config.observation_n_classes:
                    fitted = values[:, index].to(device="cpu", dtype=torch.float64)
                    statistics[column] = {
                        "mean": float(fitted.mean()),
                        "std": float(fitted.std(unbiased=False).clamp_min(1e-6)),
                    }
        self.set_normalization_statistics(statistics)

    def model_inputs(
        self, inputs: PolicyInputs
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Translate aligned raw histories into standard named model tensors."""
        self.validate_inputs(inputs)
        features = {
            column: inputs.observations[:, :, index]
            for index, column in enumerate(self.config.observation_columns)
        }
        features.update(
            {
                column: inputs.static_inputs[:, index, None].expand_as(
                    inputs.previous_actions
                )
                for index, column in enumerate(self.config.static_columns)
            }
        )
        for column, count in self.config.observation_n_classes.items():
            values = features[column]
            if (
                (values != values.round()).any()
                or (values < 0).any()
                or (values >= count).any()
            ):
                raise ValueError(
                    f"{column}: observations must be categorical IDs in [0, {count})"
                )
            features[column] = values.long()
        for column, statistics in self.interface.selected_columns_statistics.items():
            features[column] = (features[column] - statistics["mean"]) / (
                statistics["std"] + 1e-9
            )
        features[PREVIOUS_ACTION] = inputs.previous_actions
        if self.config.return_conditioned:
            assert inputs.desired_returns is not None
            features[DESIRED_RETURN] = (
                inputs.desired_returns / self.config.returns.scale
            )
        return features, {"attention_valid_mask": inputs.attention_valid_mask}

    def forward(self, inputs: PolicyInputs) -> Tensor:
        features, metadata = self.model_inputs(inputs)
        logits = self.network(
            features, metadata, interface_name=self.interface_name
        ).logits[self.interface.target_columns[0]]
        return logits.masked_fill(~inputs.attention_valid_mask[:, :, None], 0.0)

    def distribution(self, inputs: PolicyInputs) -> CategoricalActionDistribution:
        return CategoricalActionDistribution(self(inputs))

    def validate_inputs(self, inputs: PolicyInputs) -> None:
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
        if inputs.observations.shape != (
            *mask.shape,
            len(self.config.observation_columns),
        ):
            raise ValueError("policy observation shape does not match its schema")
        if inputs.static_inputs.shape != (
            mask.shape[0],
            len(self.config.static_columns),
        ):
            raise ValueError("policy static input shape does not match its schema")
        actions = inputs.previous_actions
        if (
            actions.shape != mask.shape
            or actions.dtype != torch.int64
            or (actions < 0).any()
            or (actions > len(self.config.action_values)).any()
        ):
            raise ValueError(
                "policy previous actions must be valid int64 IDs, including the start token"
            )
        numeric = [inputs.observations, inputs.static_inputs]
        if self.config.return_conditioned:
            if (
                inputs.desired_returns is None
                or inputs.desired_returns.shape != mask.shape
            ):
                raise ValueError(
                    "policy requires a desired return at every decision position"
                )
            numeric.append(inputs.desired_returns)
        if any(
            value.device != self.device
            or value.dtype != torch.float32
            or not torch.isfinite(value).all()
            for value in numeric
        ):
            raise ValueError(
                "numeric policy inputs must be finite float32 tensors on the model device"
            )
        if mask.device != self.device or actions.device != self.device:
            raise ValueError("policy masks and actions must be on the model device")
