"""Masked child encoding, preserving physical slots and a CLS-only empty path."""

import torch
from torch import nn

from sequifier.config.depth_layout import depth_mask_metadata_key
from sequifier.model.backbone import TransformerBackbone
from sequifier.model.dtypes import cast_floating_to_module_dtype, module_param_dtype
from sequifier.model.encoder_stack import TransformerEncoderStack
from sequifier.model.ingestions import (
    BaseFeatureIngestion,
    RealFeatureProjection,
    embedding_safe_indices,
    get_feature_embedding_dims,
)
from sequifier.special_tokens import SPECIAL_TOKEN_IDS


class DepthTransformerIngestion(BaseFeatureIngestion):
    def __init__(
        self,
        *,
        config,
        layout,
        categorical_columns,
        real_columns,
        n_classes,
        context_length,
        add_ingestion_position,
    ):
        super().__init__()
        self.config = config
        self.layout = layout
        self.columns = tuple(config.columns)
        self.categorical_columns = frozenset(categorical_columns)
        self.mask_key = depth_mask_metadata_key(config.layout)
        architecture = config.architecture
        width = architecture.dim_model
        dims = config.feature_embedding_dims or get_feature_embedding_dims(
            width, categorical_columns, real_columns
        )
        self.feature_dims = dims
        self.encoder = nn.ModuleDict()
        for column in self.columns:
            if column in self.categorical_columns:
                if not 0 <= SPECIAL_TOKEN_IDS.unknown < n_classes[column]:
                    raise ValueError(f"No legal padding token for {column!r}")
                self.encoder[column] = nn.Embedding(n_classes[column], dims[column])
            else:
                self.encoder[column] = RealFeatureProjection(dims[column])
        self.input_projection = (
            nn.Linear(sum(dims.values()), width)
            if sum(dims.values()) != width
            else nn.Identity()
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.position_embedding = (
            nn.Embedding(layout.context_length + 1, width)
            if architecture.position_encoding.type == "learned"
            else None
        )
        self.register_buffer(
            "sinusoidal_positions",
            TransformerBackbone._sinusoidal_positions(
                layout.context_length + 1,
                width,
                architecture.position_encoding.theta,
            ),
            persistent=False,
        )
        self.position_dropout = nn.Dropout(architecture.dropout)
        self.stack = TransformerEncoderStack(architecture, layout.context_length + 1)
        self.output_projection_layer = (
            nn.Linear(width, config.output_dim)
            if width != config.output_dim
            else nn.Identity()
        )
        self.pos_encoder = (
            nn.Embedding(context_length, config.output_dim)
            if add_ingestion_position
            else None
        )
        self.drop = nn.Dropout(config.dropout)
        # Ownership affects new optimizer IDs without changing legacy attention ingestions.
        for parameter in self.parameters():
            parameter._sequifier_depth_parameter = True

    def forward(self, src, metadata):
        valid = metadata[self.mask_key]
        encoded = []
        for column in self.columns:
            raw = src[column]
            categorical = column in self.categorical_columns
            safe = torch.where(
                valid,
                raw,
                torch.full_like(raw, SPECIAL_TOKEN_IDS.unknown if categorical else 0),
            )
            module = self.encoder[column]
            if categorical:
                value = (
                    module(embedding_safe_indices(safe))
                    * self.feature_dims[column] ** 0.5
                )
            else:
                value = module(
                    cast_floating_to_module_dtype(safe.unsqueeze(-1), module)
                )
            encoded.append(value)
        dtype = module_param_dtype(self.input_projection) or encoded[0].dtype
        cells = torch.cat([value.to(dtype=dtype) for value in encoded], dim=-1)
        cells = self.input_projection(cells)
        batch, time, depth, width = cells.shape
        cells = cells.reshape(batch * time, depth, width)
        cls = self.cls.to(dtype=cells.dtype).expand(batch * time, -1, -1)
        cells = torch.cat((cls, cells), dim=1)
        if self.position_embedding is not None:
            positions = self.position_embedding(
                torch.arange(depth + 1, device=cells.device)
            )
            cells = cells + positions.to(dtype=cells.dtype).unsqueeze(0)
        elif self.config.architecture.position_encoding.type == "sinusoidal":
            cells = cells + self.sinusoidal_positions[: depth + 1].to(dtype=cells.dtype)
        cells = self.position_dropout(cells)
        keys = torch.cat(
            (
                torch.ones((batch * time, 1), dtype=torch.bool, device=valid.device),
                valid.reshape(batch * time, depth),
            ),
            dim=1,
        )
        output = self.stack(cells, keys[:, None, None, :])[:, 0]
        output = self.output_projection_layer(
            cast_floating_to_module_dtype(output, self.output_projection_layer)
        )
        output = output.reshape(batch, time, self.config.output_dim)
        if self.pos_encoder is not None:
            positions = self.pos_encoder(torch.arange(time, device=output.device))
            output = output + positions.to(dtype=output.dtype).unsqueeze(0)
        return self.drop(output)
