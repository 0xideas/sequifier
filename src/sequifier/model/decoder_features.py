"""Expand backbone support windows into output feature representations."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sequifier.model.dtypes import cast_floating_to_module_dtype
from sequifier.model.encoder_stack import TransformerEncoderStack


class FeatureReadout(nn.Module):
    """Condition learned feature tokens on the same temporal context."""

    def __init__(self, input_dim: int, width: int, targets: list[str]):
        super().__init__()
        self.context = nn.Linear(input_dim, width)
        self.feature_tokens = nn.Embedding(len(targets), width)
        self.target_indices = {name: i for i, name in enumerate(targets)}

    def forward(self, x: Tensor, targets):
        context = self.context(cast_floating_to_module_dtype(x, self.context))
        return {
            name: F.gelu(
                context
                + self.feature_tokens.weight[self.target_indices[name]].to(
                    context.dtype
                )
            )
            for name in targets
        }


class GroupedReadout(nn.Module):
    def __init__(self, input_dim, width, groups):
        super().__init__()
        self.groups = groups
        self.readouts = nn.ModuleDict(
            {
                name: FeatureReadout(input_dim, width, targets)
                for name, targets in groups.items()
            }
        )

    def forward(self, x, targets):
        result = {}
        for group, members in self.groups.items():
            selected = [target for target in members if target in targets]
            if selected:
                result.update(self.readouts[group](x, selected))
        return result


class TemporalConvReadout(nn.Module):
    """Convolve within each causal support window, ending at its prediction."""

    def __init__(self, model_dim, support_length, config):
        super().__init__()
        self.support_length = support_length
        self.model_dim = model_dim
        self.conv = nn.Conv1d(
            model_dim, config.channels, config.kernel_size, dilation=config.dilation
        )
        self.receptive_field = 1 + (config.kernel_size - 1) * config.dilation
        if self.receptive_field > support_length:
            raise ValueError(
                "temporal_conv decoder receptive field exceeds support_length"
            )

    def forward(self, x, targets):
        history = x.reshape(-1, self.support_length, self.model_dim).transpose(1, 2)
        history = history[:, :, -self.receptive_field :]
        token = F.gelu(self.conv(cast_floating_to_module_dtype(history, self.conv)))
        token = token.reshape(*x.shape[:-1], self.conv.out_channels)
        return dict.fromkeys(targets, token)


class StaticReadout(nn.Module):
    """Project the sequence summary back into per-attribute embedding blocks."""

    def __init__(self, input_dim, width, targets):
        super().__init__()
        self.width = width
        self.projection = nn.Linear(input_dim, width * len(targets))
        self.target_indices = {name: i for i, name in enumerate(targets)}

    def forward(self, x, targets):
        summary = x[:, -1:]
        features = self.projection(
            cast_floating_to_module_dtype(summary, self.projection)
        )
        blocks = F.gelu(features).split(self.width, dim=-1)
        return {name: blocks[self.target_indices[name]] for name in targets}


class DepthReadout(nn.Module):
    """Seed one learned query per physical slot and jointly refine the children."""

    def __init__(self, input_dim, layout, config):
        super().__init__()
        width = config.architecture.model_dim
        self.context = nn.Linear(input_dim, width)
        if config.architecture.position_encoding.type == "sinusoidal":
            from sequifier.model.backbone import TransformerBackbone

            self.slot_queries = None
            self.register_buffer(
                "slot_positions",
                TransformerBackbone._sinusoidal_positions(
                    layout.max_depth_length,
                    width,
                    config.architecture.position_encoding.theta,
                ),
                persistent=False,
            )
        else:
            self.slot_queries = nn.Embedding(layout.max_depth_length, width)
        self.stack = TransformerEncoderStack(
            config.architecture, layout.max_depth_length
        )
        self.occupancy = nn.Linear(width, 1)
        self.allow_gaps = layout.allow_gaps
        self.occupancy_key = f"depth_occupancy:{config.layout}"

    def forward(self, x, targets):
        context = self.context(cast_floating_to_module_dtype(x, self.context))
        positions = (
            self.slot_positions
            if self.slot_queries is None
            else self.slot_queries.weight
        )
        slots = context.unsqueeze(-2) + positions.to(context.dtype)
        hidden = self.stack(slots.flatten(0, 1)).reshape_as(slots)
        result = dict.fromkeys(targets, hidden)
        scores = self.occupancy(
            cast_floating_to_module_dtype(hidden, self.occupancy)
        ).float()
        if not self.allow_gaps:
            # Predict a distribution over lengths 0..capacity. The last length
            # is the reference category, so each slot supplies one free logit.
            lengths = torch.cat((scores, torch.zeros_like(scores[..., :1, :])), dim=-2)
            mass = lengths.softmax(dim=-2)
            before = mass[..., :-1, :].cumsum(dim=-2)
            after = mass[..., 1:, :].flip((-2,)).cumsum(dim=-2).flip((-2,))
            scores = after.clamp_min(1e-30).log() - before.clamp_min(1e-30).log()
        result[self.occupancy_key] = scores
        return result
