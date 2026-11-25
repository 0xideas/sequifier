"""Independent history value and state-action critics for offline IQL."""

from contextlib import nullcontext
from copy import deepcopy

import torch
from torch import nn

from sequifier.io.batch import SequifierInputs
from sequifier.io.policy_window import normalize_policy_features
from sequifier.model.dtypes import cast_floating_to_module_dtype
from sequifier.model.factory import build_history_encoder
from sequifier.model.history_transfer import copy_history_weights


def critic_history_options(actor, phase, *, metadata_only=False, restoring=False):
    if metadata_only:
        return {"device": torch.device("meta")}
    restore = restoring or phase.optimizer_state == "continue"
    return {
        "device": actor.device,
        "history_source": actor.network
        if phase.history_initialization == "phase_start_actor" and not restore
        else None,
        "restore_history": restore,
    }


def validate_independent_networks(*modules):
    seen = set()
    for module in modules:
        storage = {
            (tensor.device, tensor.untyped_storage().data_ptr())
            for tensor in (*module.parameters(), *module.buffers())
            if tensor.numel()
        }
        if seen & storage:
            raise ValueError("policy networks share parameter or buffer storage")
        seen.update(storage)


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
                    actions[predecessor], actions[predecessor].new_tensor(allowed)
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
        self,
        execution_config,
        interface_name: str,
        action_count: int | dict | None,
        *,
        history_source=None,
        restore_history=False,
        device: torch.device,
    ):
        super().__init__()
        self.interface_name = interface_name
        self.interface = execution_config.interface(interface_name)
        with device if device.type == "meta" else nullcontext():
            self.network = build_history_encoder(
                execution_config,
                interface_name,
                device=device,
                restored_history=restore_history or history_source is not None,
            )
            width = self.network.model_dim
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
        self.to(device)
        if history_source is not None:
            copy_history_weights(history_source, self.network, interface_name)

    @classmethod
    def ppo_value(cls, actor, **history):
        value = cls(actor.execution_config, actor.interface_name, None, **history)
        if history.get("history_source") is not None:
            validate_independent_networks(actor.network, value)
        return value

    def forward(
        self,
        history: SequifierInputs,
        action: torch.Tensor | dict[str, torch.Tensor] | None = None,
    ):
        if (action is None) != (self.action_embedding is None):
            raise ValueError("Q requires a current action and V forbids one")
        features = normalize_policy_features(history, self.interface)
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
        return (
            self.head(cast_floating_to_module_dtype(state, self.head[0]))
            .squeeze(-1)
            .float()
        )


def make_iql_critics(
    execution_config, interface_name: str, action_count: int | dict, **history
):
    value = HistoryCritic(execution_config, interface_name, None, **history)
    q1 = HistoryCritic(execution_config, interface_name, action_count, **history)
    q2 = HistoryCritic(execution_config, interface_name, action_count, **history)
    return value, q1, q2


def make_iql_models(
    execution_config, interface_name: str, action_count: int | dict, **history
):
    value, q1, q2 = make_iql_critics(
        execution_config, interface_name, action_count, **history
    )
    targets = (
        deepcopy(q1).requires_grad_(False),
        deepcopy(q2).requires_grad_(False),
    )
    modules = (value, q1, q2, *targets)
    if len(
        {id(parameter) for module in modules for parameter in module.parameters()}
    ) != sum(1 for module in modules for _ in module.parameters()):
        raise ValueError("IQL networks share parameters")
    if (source := history.get("history_source")) is not None:
        validate_independent_networks(source, *modules)
    return value, q1, q2, targets
