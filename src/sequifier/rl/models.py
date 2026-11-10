"""Independent history value and state-action critics for offline IQL."""

from copy import deepcopy

import torch
from torch import nn

from sequifier.io.batch import SequifierInputs
from sequifier.io.policy_window import normalize_policy_features
from sequifier.model.factory import build_transformer_network


class ActionEncoder(nn.Module):
    """Fuse typed categorical embeddings and standardized real projections."""

    def __init__(self, fields, width, statistics):
        super().__init__()
        self.fields = {name: fields[name] for name in sorted(fields)}
        self.statistics = statistics
        self.encoders = nn.ModuleDict(
            {
                name: nn.Embedding(len(field.values), width)
                if field.type == "categorical"
                else nn.Linear(1, width)
                for name, field in self.fields.items()
            }
        )
        self.fusion = nn.Linear(len(fields) * (width + 1), width)

    def forward(self, actions):
        from sequifier.io.preprocessing import normalize_values

        pieces = []
        for name, field in self.fields.items():
            value = actions[name]
            active = torch.ones_like(value, dtype=torch.bool)
            for predecessor, labels in field.active_when.items():
                parent = self.fields[predecessor]
                allowed = (
                    [parent.values.index(label) for label in labels]
                    if parent.type == "categorical"
                    else labels
                )
                active &= torch.isin(
                    actions[predecessor], torch.tensor(allowed, device=value.device)
                )
            if field.type == "real":
                value = value.float()
                if name in self.statistics:
                    value = normalize_values(value, self.statistics[name])
                value = value[:, None]
            encoded = self.encoders[name](value)
            pieces.extend((encoded * active[:, None], active.float()[:, None]))
        return self.fusion(torch.cat(pieces, dim=-1))


class HistoryCritic(nn.Module):
    def __init__(
        self, execution_config, interface_name: str, action_count: int | dict | None
    ):
        super().__init__()
        self.network = build_transformer_network(
            execution_config, device=torch.device(execution_config.device)
        ).network
        self.interface_name = interface_name
        self.interface = execution_config.interface(interface_name)
        for parameter in self.network.resolve_interface(
            interface_name
        ).decoder.parameters():
            parameter.requires_grad_(False)
        width = self.network.dim_model
        self.action_embedding = (
            ActionEncoder(
                action_count,
                width,
                self.interface.selected_columns_statistics
                if self.interface.normalize_real_columns
                else {},
            )
            if isinstance(action_count, dict)
            else nn.Embedding(action_count, width)
            if action_count
            else None
        )
        self.head = nn.Sequential(
            nn.Linear(width * (2 if action_count else 1), width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )
        self.to(execution_config.device)

    def forward(
        self,
        history: SequifierInputs,
        action: torch.Tensor | dict[str, torch.Tensor] | None = None,
    ):
        if (action is None) != (self.action_embedding is None):
            raise ValueError("Q requires a current action and V forbids one")
        features = normalize_policy_features(
            history.inputs, history.metadata, self.interface
        )
        return self.prepared_value(features, history.metadata, action)

    def prepared_value(self, features, metadata, action=None):
        """Score the same normalized history used by the collecting actor."""
        representation = self.network.encode(
            features, metadata, interface_name=self.interface_name
        )
        state = representation[:, -1]
        if self.action_embedding is not None:
            if action is None:
                raise ValueError("Q requires a current action")
            if not isinstance(action, dict) and (
                action.dtype != torch.long
                or action.ndim != 1
                or action.shape[0] != state.shape[0]
            ):
                raise ValueError("critic actions must be int64 [batch]")
            state = torch.cat((state, self.action_embedding(action)), dim=-1)
        return self.head(state).squeeze(-1).float()


def make_iql_models(execution_config, interface_name: str, action_count: int | dict):
    value = HistoryCritic(execution_config, interface_name, None)
    q1 = HistoryCritic(execution_config, interface_name, action_count)
    q2 = HistoryCritic(execution_config, interface_name, action_count)
    targets = (
        deepcopy(q1).requires_grad_(False),
        deepcopy(q2).requires_grad_(False),
    )
    modules = (value, q1, q2, *targets)
    if len(
        {id(parameter) for module in modules for parameter in module.parameters()}
    ) != sum(1 for module in modules for _ in module.parameters()):
        raise ValueError("IQL networks share parameters")
    return value, q1, q2, targets
