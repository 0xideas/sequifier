"""A small causal numeric actor with one categorical head and no value network."""

import torch
from torch import Tensor, nn

from sequifier.config.policy_config import CategoricalPolicyConfig
from sequifier.rl.distributions import CategoricalActionDistribution
from sequifier.rl.history import PolicyInputs


class CategoricalPolicy(nn.Module):
    def __init__(self, config: CategoricalPolicyConfig):
        super().__init__()
        self.config = config
        width = config.hidden_size
        numeric_width = len(config.observation_columns) + len(config.static_columns) + 1
        self.ingestion = nn.Linear(numeric_width, width)
        self.previous_action_embedding = nn.Embedding(
            len(config.action_values) + 1, width
        )
        self.backbone = nn.GRUCell(width, width)
        self.decoder = nn.Linear(width, len(config.action_values))
        self.register_buffer(
            "observation_mean", torch.zeros(len(config.observation_columns))
        )
        self.register_buffer(
            "observation_scale", torch.ones(len(config.observation_columns))
        )
        self.register_buffer("static_mean", torch.zeros(len(config.static_columns)))
        self.register_buffer("static_scale", torch.ones(len(config.static_columns)))

    def set_normalization(self, observations: Tensor, static_inputs: Tensor) -> None:
        """Fit once on the training episode partition; persist as model buffers."""
        for values, mean, scale in (
            (observations, self.observation_mean, self.observation_scale),
            (static_inputs, self.static_mean, self.static_scale),
        ):
            if (
                values.ndim != 2
                or values.shape[1:] != mean.shape
                or not len(values)
                or not torch.isfinite(values).all()
            ):
                raise ValueError(
                    "normalization requires finite training rows matching policy columns"
                )
            fitted = values.to(torch.float64)
            mean.copy_(fitted.mean(dim=0))
            scale.copy_(fitted.std(dim=0, unbiased=False).clamp_min(1e-6))
            if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
                raise ValueError(
                    "training normalization exceeds the model's numeric range"
                )

    def forward(self, inputs: PolicyInputs) -> Tensor:
        self.validate_inputs(inputs)
        mask = inputs.attention_valid_mask
        batch, context = mask.shape
        observations = (
            inputs.observations - self.observation_mean
        ) / self.observation_scale
        static = (inputs.static_inputs - self.static_mean) / self.static_scale
        static = static[:, None, :].expand(batch, context, -1)
        if self.config.return_conditioned:
            assert inputs.desired_returns is not None
            returns = inputs.desired_returns / self.config.returns.scale
        else:
            # Same parameters and architecture as RC; the return channel is disabled.
            returns = observations.new_zeros((batch, context))
        numeric = torch.cat((observations, static, returns.unsqueeze(-1)), dim=-1)
        encoded = torch.tanh(
            self.ingestion(numeric)
            + self.previous_action_embedding(inputs.previous_actions)
        )
        hidden = encoded.new_zeros((batch, self.config.hidden_size))
        outputs = []
        for position in range(context):
            candidate = self.backbone(encoded[:, position], hidden)
            hidden = torch.where(mask[:, position, None], candidate, hidden)
            outputs.append(self.decoder(hidden))
        return torch.stack(outputs, dim=1)

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
            value.device.type != "cpu"
            or value.dtype != torch.float32
            or not torch.isfinite(value).all()
            for value in numeric
        ):
            raise ValueError("numeric policy inputs must be finite CPU float32 tensors")
        if mask.device.type != "cpu" or actions.device.type != "cpu":
            raise ValueError("numeric policies currently support CPU execution only")
