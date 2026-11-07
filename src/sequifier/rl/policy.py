"""Policy alignment adapter over Sequifier's configurable model system."""

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from sequifier.artifacts.model_config import execution_config
from sequifier.config.metadata import DatasetMetadata
from sequifier.config.policy_config import CategoricalPolicyConfig, PolicySchemaConfig
from sequifier.config.train_config import GlobalTrainingSpecModel, _resolve_interface
from sequifier.io.batch import SequifierInputs
from sequifier.io.policy_window import (
    DESIRED_RETURN,
    PREVIOUS_ACTION,
    action_global_ids,
    action_target,
    normalize_policy_features,
)
from sequifier.io.preprocessing import normalization_statistics
from sequifier.model.execution_schema import ExecutionSchema
from sequifier.rl.distributions import CategoricalActionDistribution
from sequifier.special_tokens import SPECIAL_TOKEN_IDS, SPECIAL_TOKEN_NAMES


def policy_execution_config(
    config: CategoricalPolicyConfig,
    model,
    device: str = "cpu",
    *,
    global_training=None,
    statistics=None,
    **run_options,
):
    """Resolve policy alignment directly into the ordinary execution contract."""
    columns = config.observation_columns + config.static_columns + [PREVIOUS_ACTION]
    if config.return_conditioned:
        columns.append(DESIRED_RETURN)
    target = action_target(config)
    categorical = list(config.categorical_n_classes) + [PREVIOUS_ACTION, target]
    real = [column for column in columns if column not in categorical]
    decoder_ids = action_global_ids(config)
    action_width = decoder_ids[-1] + 1
    global_training = global_training or GlobalTrainingSpecModel(
        training_objective="policy",
        context_length=config.context_length,
        target_offset=0,
        inference_batch_size=1,
        batch_size=1,
        learning_rate=1e-3,
        torch_compile="none",
    )
    interface = model.interfaces[config.interface_name].model_copy(
        update={
            "input_columns": columns,
            "target_columns": [target],
            "policy": PolicySchemaConfig.model_validate(
                config.model_dump(include=set(PolicySchemaConfig.model_fields))
            ),
        }
    )
    metadata = DatasetMetadata(
        window_length=config.context_length + 1,
        column_data_types={
            column: "Int64" if column in categorical else "Float32"
            for column in columns + [target]
        },
        n_classes={
            **config.categorical_n_classes,
            PREVIOUS_ACTION: action_width + 1,
            target: action_width,
        },
        id_maps={
            **config.id_maps,
            target: {
                str(value): global_id
                for value, global_id in zip(config.action_values, decoder_ids)
            },
        },
        selected_columns_statistics=statistics
        if statistics is not None
        else {
            column: {"mean": 0.0, "std": 1.0}
            for column in real
            if column != DESIRED_RETURN
        },
    )
    # Legacy action IDs can occupy reserved-token slots. Resolve the same IDs
    # through the ordinary codec, then retain the persisted policy contract.
    spec = interface.model_copy(
        update={
            "categorical_decoder_special_tokens": {
                target: [
                    name
                    for name in SPECIAL_TOKEN_NAMES
                    if getattr(SPECIAL_TOKEN_IDS, name) in decoder_ids
                ]
            }
        }
    )
    resolved = _resolve_interface(
        config.interface_name, spec, metadata, global_training, metadata.storage_layout
    )
    resolved.policy = config
    resolved.categorical_columns = categorical
    resolved.categorical_decoder_special_tokens = {}
    resolved.selected_columns_statistics = metadata.selected_columns_statistics
    return execution_config(
        model.model_copy(update={"interfaces": {config.interface_name: interface}}),
        {config.interface_name: resolved},
        global_training,
        device=device,
        **run_options,
    )


class CategoricalPolicy(nn.Module):
    """A convenience view; the network owns all trainable weights and execution."""

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

    def set_normalization_statistics(
        self, statistics: Mapping[str, Mapping[str, float]]
    ) -> None:
        expected = (
            set(self.config.observation_columns + self.config.static_columns)
            - self.config.categorical_n_classes.keys()
        )
        if set(statistics) != expected:
            raise ValueError("normalization statistics must match real policy columns")
        self.interface.selected_columns_statistics = normalization_statistics(
            {
                column: {key: float(values[key]) for key in ("mean", "std")}
                for column, values in statistics.items()
            }
        )

    def model_inputs(
        self, inputs: SequifierInputs
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Encode raw named histories without mutating stored data or masks."""
        self.validate_inputs(inputs)
        return normalize_policy_features(
            inputs.inputs, inputs.metadata, self.interface
        ), inputs.metadata

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
