"""Encode per-sequence scalar features as one temporal prefix token."""

import math

import torch
from torch import Tensor, nn

from sequifier.model.dtypes import cast_floating_to_module_dtype
from sequifier.model.ingestions import get_feature_embedding_dims


class StaticConditionEncoder(nn.Module):
    def __init__(
        self,
        columns: list[str],
        categorical_columns: list[str],
        real_columns: list[str],
        n_classes: dict[str, int],
        model_dim: int,
        total_embedding_dim: int | None = None,
        feature_embedding_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.columns = tuple(columns)
        self.categorical_columns = tuple(categorical_columns)
        self.real_columns = tuple(real_columns)
        if feature_embedding_dims is not None:
            widths = feature_embedding_dims
        elif total_embedding_dim is not None:
            widths = get_feature_embedding_dims(
                total_embedding_dim, categorical_columns, real_columns, n_classes
            )
        else:
            widths = {
                column: max(4, min(32, math.isqrt(n_classes[column])))
                for column in categorical_columns
            }
            widths.update({column: 1 for column in real_columns})
        if set(widths) != set(columns):
            raise ValueError(
                "Every static conditioning column needs an embedding width"
            )
        self.embeddings = nn.ModuleList(
            nn.Embedding(n_classes[column], widths[column])
            for column in categorical_columns
        )
        self.real_projections = nn.ModuleList(
            nn.Identity() if widths[column] == 1 else nn.Linear(1, widths[column])
            for column in real_columns
        )
        input_width = sum(widths.values())
        if input_width == 0 or set(columns) != set(categorical_columns + real_columns):
            raise ValueError(
                "Every static conditioning column must be categorical or real"
            )
        self.projection = nn.Linear(input_width, model_dim)
        self.projection._sequifier_layer_group = "ingestion.output_projection"

    def forward(self, features: dict[str, Tensor]) -> Tensor:
        parts = [
            embedding(features[column].to(torch.int64))
            for column, embedding in zip(self.categorical_columns, self.embeddings)
        ]
        parts.extend(
            projection(
                cast_floating_to_module_dtype(
                    features[column].unsqueeze(-1).float(), projection
                )
            )
            for column, projection in zip(self.real_columns, self.real_projections)
        )
        combined = torch.cat(parts, dim=-1)
        return self.projection(
            cast_floating_to_module_dtype(combined, self.projection)
        ).unsqueeze(1)
