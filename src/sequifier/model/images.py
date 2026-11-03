"""Image encoders and view-level ingestion."""

from typing import Any

import torch
from torch import Tensor, nn

from sequifier.model.ingestions import BaseFeatureIngestion, _AxisPoolBlock


class ImageEncoder(nn.Module):
    def __init__(self, config: Any, output_dim: int, load_pretrained: bool = True):
        super().__init__()
        try:
            from torchvision import models
            from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
        except ImportError as error:
            raise ImportError(
                "Image ingestion requires the sequifier[images] extra"
            ) from error

        self.frozen = config.type == "frozen"
        if self.frozen:
            weights_type = models.get_model_weights(config.model)
            try:
                weights = weights_type[config.weights]
            except KeyError as error:
                raise ValueError(
                    f"Unknown weights {config.weights!r} for {config.model}"
                ) from error
            self.backbone = models.get_model(
                config.model, weights=weights if load_pretrained else None
            )
        else:
            depths = {
                18: (BasicBlock, [2, 2, 2, 2]),
                34: (BasicBlock, [3, 4, 6, 3]),
                50: (Bottleneck, [3, 4, 6, 3]),
                101: (Bottleneck, [3, 4, 23, 3]),
                152: (Bottleneck, [3, 8, 36, 3]),
            }
            block, layers = depths[config.depth]
            if config.layers is not None:
                block = BasicBlock if config.block == "basic" else Bottleneck
                layers = list(config.layers)
            self.backbone = ResNet(
                block,
                layers,
                groups=config.groups,
                width_per_group=config.width_per_group,
                replace_stride_with_dilation=list(config.replace_stride_with_dilation),
            )
        width = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection = nn.Linear(width, output_dim)
        self.output_dim = output_dim
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        if self.frozen:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def _encode(self, images: Tensor) -> Tensor:
        images = images.float() / 255.0
        normalized = (images - self.mean) / self.std
        normalized = normalized.to(dtype=next(self.backbone.parameters()).dtype)
        if self.frozen:
            with torch.no_grad():
                encoded = self.backbone(normalized)
        else:
            encoded = self.backbone(normalized)
        return self.projection(encoded.to(dtype=self.projection.weight.dtype))

    def forward(self, images: Tensor, valid: Tensor | None = None) -> Tensor:
        shape = images.shape[:-3]
        flat = images.reshape(-1, *images.shape[-3:])
        if self.training and valid is not None and not torch.compiler.is_compiling():
            indices = valid.reshape(-1).nonzero().flatten()
            if indices.numel() == 0:
                return self.projection.weight.new_zeros(*shape, self.output_dim)
            selected = self._encode(flat.index_select(0, indices))
            output = selected.new_zeros(flat.shape[0], self.output_dim)
            return output.index_copy(0, indices, selected).reshape(*shape, -1)
        return self._encode(flat).reshape(*shape, -1)


class ImageViewEncoder(nn.Module):
    def __init__(
        self,
        views: list[str],
        encoder_config: Any,
        output_dim: int,
        frame_pool: str,
        load_pretrained: bool = True,
    ):
        super().__init__()
        self.views = tuple(views)
        self.encoder = ImageEncoder(encoder_config, output_dim, load_pretrained)
        self.view_embedding = nn.Parameter(torch.zeros(len(views), output_dim))
        self.frame_score = (
            nn.Linear(output_dim, 1) if frame_pool == "attention" else None
        )
        self.frame_sum = _AxisPoolBlock(
            axes=["frame"], mode="sum", active_axes=["frame"]
        )

    def forward(
        self, inputs: dict[str, Tensor], metadata: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        output = {}
        for index, view in enumerate(self.views):
            images = inputs[f"__image:{view}"]
            valid = metadata[f"__image_mask:{view}"].bool()
            encoded = self.encoder(images, valid)
            frame_score = self.frame_score
            if frame_score is None:
                weights = valid.to(encoded.dtype)
                pooled = (
                    self.frame_sum(encoded * weights[..., None])
                    / weights.sum(dim=2).clamp(min=1)[..., None]
                )
            else:
                scores = frame_score(encoded).squeeze(-1)
                scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
                weights = torch.softmax(scores, dim=2) * valid
                weights = weights / weights.sum(dim=2, keepdim=True).clamp(min=1e-12)
                pooled = (encoded * weights[..., None]).sum(dim=2)
            output[view] = pooled + self.view_embedding[index]
        return output


class ImagePrefixEncoder(nn.Module):
    def __init__(
        self, config: Any, output_dim: int, view_count: int, load_pretrained: bool
    ):
        super().__init__()
        self.encoder = ImageEncoder(config, output_dim, load_pretrained)
        self.merge = (
            nn.Linear(view_count * output_dim, output_dim) if view_count > 1 else None
        )

    def forward(self, images: Tensor) -> Tensor:
        encoded = self.encoder(images)
        if self.merge is None:
            return encoded[:, 0]
        return self.merge(encoded.flatten(start_dim=1))


class ImageFeatureIngestion(BaseFeatureIngestion):
    def __init__(
        self,
        config: Any,
        context_length: int,
        add_ingestion_position: bool,
        load_pretrained: bool = True,
    ):
        super().__init__()
        self.views = tuple(config.views)
        self.encoder = ImageViewEncoder(
            config.views,
            config.encoder,
            config.output_dim,
            config.frame_pool,
            load_pretrained,
        )
        self.merge = (
            nn.Linear(len(config.views) * config.output_dim, config.output_dim)
            if config.view_merge == "concat" and len(config.views) > 1
            else None
        )
        self.add_ingestion_position = add_ingestion_position
        self.position = (
            nn.Embedding(context_length, config.output_dim)
            if add_ingestion_position
            else None
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]) -> Tensor:
        views = self.encoder(src, metadata)
        if self.merge is not None:
            output = self.merge(torch.cat([views[name] for name in self.views], dim=-1))
        else:
            output = torch.stack([views[name] for name in self.views], dim=0).mean(
                dim=0
            )
        if self.position is not None:
            positions = torch.arange(output.shape[1], device=output.device)
            output = output + self.position(positions)[None]
        return self.dropout(output)
