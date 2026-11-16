"""Image encoders and view-level ingestion."""

import math
from typing import Any

import torch
from torch import Tensor, nn

from sequifier.io.image_feature_cache import IMAGE_CACHE_PREFIX
from sequifier.model.dtypes import cast_floating_to_module_dtype
from sequifier.model.frozen_image_cache import FrozenImageCache, encoder_signature
from sequifier.model.ingestions import BaseFeatureIngestion, _AxisPoolBlock


def _per_image_norm(channels: int) -> nn.GroupNorm:
    """Normalize each frame independently of other frames in the window."""
    return nn.GroupNorm(math.gcd(32, channels), channels)


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
        self.microbatch_size = config.microbatch_size
        self._feature_cache: FrozenImageCache | None = None
        self._cache_signature = None
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
                norm_layer=_per_image_norm,
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

    def clear_feature_cache(self):
        """Invalidate transient features after an external frozen-weight change."""
        if self._feature_cache is not None:
            self._feature_cache = FrozenImageCache(self._feature_cache.budget)
        self._cache_signature = None

    def _backbone_microbatch(self, images: Tensor) -> Tensor:
        images = images.float() / 255.0
        normalized = (images - self.mean) / self.std
        normalized = normalized.to(dtype=next(self.backbone.parameters()).dtype)
        if self.frozen:
            with torch.no_grad():
                encoded = self.backbone(normalized)
        else:
            encoded = self.backbone(normalized)
        return encoded

    def _encode_microbatch(self, images: Tensor) -> Tensor:
        return self.projection(
            self._backbone_microbatch(images).to(dtype=self.projection.weight.dtype)
        )

    def _encode(self, images: Tensor) -> Tensor:
        if torch.onnx.is_in_onnx_export():
            return self._encode_microbatch(images)
        if images.shape[0] == 0:
            return self.projection.weight.new_zeros(0, self.output_dim)
        # Bound backbone call sizes independently of policy scoring batches.
        return torch.cat(
            [
                self._encode_microbatch(chunk)
                for chunk in images.split(self.microbatch_size)
            ]
        )

    @torch.compiler.disable
    def _encode_valid(self, flat: Tensor, valid: Tensor, cache_ids=None) -> Tensor:
        """Skip padded frames while preserving their positions in the output."""
        indices = valid.reshape(-1).nonzero().flatten()
        if indices.numel() == 0:
            return self.projection.weight.new_zeros(flat.shape[0], self.output_dim)
        if self._feature_cache is not None and cache_ids is not None:
            signature = encoder_signature(self)
            if signature is not None:
                if signature != self._cache_signature:
                    self.clear_feature_cache()
                    self._cache_signature = signature
                return self._encode_cached(flat, indices, cache_ids)
            self.clear_feature_cache()
        # Gather one microbatch at a time, avoiding a second full image tensor.
        selected = torch.cat(
            [
                self._encode_microbatch(flat.index_select(0, chunk))
                for chunk in indices.split(self.microbatch_size)
            ]
        )
        output = selected.new_zeros(flat.shape[0], self.output_dim)
        return output.index_copy(0, indices, selected)

    def _encode_cached(self, flat, indices, cache_ids):
        cache = self._feature_cache
        assert cache is not None
        if cache_ids.numel() != flat.shape[0]:
            raise ValueError("image cache identities must match the frame axes")
        identities = cache_ids.reshape(-1).index_select(0, indices).cpu().tolist()
        positions = indices.cpu().tolist()
        if any(identity <= 0 for identity in identities):
            raise ValueError("valid images require positive transient identities")
        shape = tuple(flat.shape[1:])
        keys = [(shape, identity) for identity in identities]
        features, missing = {}, {}
        for position, key in zip(positions, keys):
            if key in features or key in missing:
                continue
            feature = cache.get(key)
            if feature is None:
                missing[key] = position
            else:
                features[key] = feature
        items = list(missing.items())
        for start in range(0, len(items), self.microbatch_size):
            chunk = items[start : start + self.microbatch_size]
            rows = torch.tensor([position for _, position in chunk], device=flat.device)
            with torch.inference_mode(False), torch.no_grad():
                encoded = self._backbone_microbatch(flat.index_select(0, rows))
            for (key, _), feature in zip(chunk, encoded):
                features[key] = feature
                cache.put(key, feature)
        # Only the frozen backbone is cached. This projection remains in autograd.
        selected = self.projection(
            torch.stack([features[key] for key in keys]).to(
                dtype=self.projection.weight.dtype
            )
        )
        output = selected.new_zeros(flat.shape[0], self.output_dim)
        return output.index_copy(0, indices, selected)

    def forward(
        self, images: Tensor, valid: Tensor | None = None, cache_ids=None
    ) -> Tensor:
        shape = images.shape[:-3]
        flat = images.reshape(-1, *images.shape[-3:])
        if (
            (self.training or self.frozen)
            and valid is not None
            and not torch.onnx.is_in_onnx_export()
        ):
            return self._encode_valid(flat, valid, cache_ids).reshape(*shape, -1)
        return self._encode(flat).reshape(*shape, -1)


class ImageViewEncoder(nn.Module):
    def __init__(
        self,
        views: list[str],
        encoder_config: Any,
        output_dim: int,
        frame_pooling: str,
        load_pretrained: bool = True,
    ):
        super().__init__()
        self.views = tuple(views)
        self.encoder = ImageEncoder(encoder_config, output_dim, load_pretrained)
        self.view_embedding = nn.Parameter(torch.zeros(len(views), output_dim))
        self.frame_score = (
            nn.Linear(output_dim, 1) if frame_pooling == "attention" else None
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
            encoded = self.encoder(
                images, valid, metadata.get(IMAGE_CACHE_PREFIX + view)
            )
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
                weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(
                    torch.finfo(weights.dtype).tiny
                )
                pooled = (encoded * weights[..., None]).sum(dim=2)
            output[view] = pooled + self.view_embedding[index].to(dtype=pooled.dtype)
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
            config.frame_pooling,
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
            combined = torch.cat([views[name] for name in self.views], dim=-1)
            output = self.merge(cast_floating_to_module_dtype(combined, self.merge))
        else:
            output = torch.stack([views[name] for name in self.views], dim=0).mean(
                dim=0
            )
        if self.position is not None:
            positions = torch.arange(output.shape[1], device=output.device)
            output = output + self.position(positions)[None]
        return self.dropout(output)
