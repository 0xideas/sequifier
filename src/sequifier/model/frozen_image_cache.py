"""Bounded, phase-scoped frozen-backbone features, outside model state dicts."""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from sequifier.io.image_feature_cache import (
    image_feature_cache_enabled,
    image_identity_scope,
)


@dataclass
class FrozenImageCache:
    budget: int
    entries: OrderedDict = field(default_factory=OrderedDict)
    retained_bytes: int = 0
    hits: int = 0
    misses: int = 0

    def __deepcopy__(self, memo):
        return type(self)(self.budget)

    def clear(self):
        self.entries.clear()
        self.retained_bytes = 0

    def get(self, key):
        feature = self.entries.get(key)
        if feature is not None:
            self.entries.move_to_end(key)
            self.hits += 1
        else:
            self.misses += 1
        return feature

    def put(self, key, feature):
        size = feature.numel() * feature.element_size()
        if size > self.budget:
            return
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.retained_bytes -= previous.numel() * previous.element_size()
        while self.retained_bytes + size > self.budget:
            _, evicted = self.entries.popitem(last=False)
            self.retained_bytes -= evicted.numel() * evicted.element_size()
        # Own a row, not a view retaining an entire microbatch's GPU storage.
        with torch.inference_mode(False), torch.no_grad():
            self.entries[key] = feature.detach().clone()
        self.retained_bytes += size


def encoder_signature(encoder):
    """Track ordinary in-place updates, weight loading and device/dtype moves.

    Inference tensors lack version counters, so caching is bypassed for them.
    Updating frozen weights via `.data` requires explicit clear_feature_cache().
    """
    if any(module.training for module in encoder.backbone.modules()) or any(
        p.requires_grad for p in encoder.backbone.parameters()
    ):
        return None
    tensors = [
        *encoder.backbone.parameters(),
        *encoder.backbone.buffers(),
        encoder.mean,
        encoder.std,
    ]
    try:
        tensor_state = tuple(
            (id(value), value.data_ptr(), value._version, value.device, value.dtype)
            for value in tensors
        )
        device_type = encoder.mean.device.type
        autocast = torch.is_autocast_enabled(device_type)
        return (
            tensor_state,
            autocast,
            torch.get_autocast_dtype(device_type) if autocast else None,
            torch.get_float32_matmul_precision(),
            torch.backends.cudnn.allow_tf32,
        )
    except RuntimeError:
        return None


def _same_backbone(left, right):
    a = {**left.backbone.state_dict(), "__mean": left.mean, "__std": left.std}
    b = {**right.backbone.state_dict(), "__mean": right.mean, "__std": right.std}
    return a.keys() == b.keys() and all(
        a[key].shape == b[key].shape
        and a[key].dtype == b[key].dtype
        and a[key].device == b[key].device
        and torch.equal(a[key], b[key])
        for key in a
    )


def _active_caches(models):
    if not image_feature_cache_enabled():
        return {}
    from sequifier.model.images import ImageEncoder

    return {
        id(encoder._feature_cache): encoder._feature_cache
        for model in models
        for encoder in model.modules()
        if isinstance(encoder, ImageEncoder) and encoder._feature_cache is not None
    }


def image_cache_statistics(*models):
    caches = _active_caches(models)
    if not caches:
        return {}
    return {
        "image_cache_hits": sum(cache.hits for cache in caches.values()),
        "image_cache_misses": sum(cache.misses for cache in caches.values()),
        "image_cache_retained_bytes": sum(
            cache.retained_bytes for cache in caches.values()
        ),
    }


def release_image_cache_entries(*models):
    # GRPO never reuses trajectories after their update. Do not retain features
    # from previous iterations merely because the phase's LRU budget has space.
    for cache in _active_caches(models).values():
        cache.clear()


@contextmanager
def frozen_image_cache_scope(actor, reference, budget):
    """Install optional GRPO caches, releasing all entries on success or failure.

    Sharing requires matching module names and exact frozen backbone state.
    Each owner checks its own tensor versions before reuse and detaches from a
    shared cache if its weights, device, dtype or normalization have changed.
    """
    from sequifier.model.images import ImageEncoder

    if not budget:
        with image_identity_scope(False):
            yield
        return
    saved, active = [], {}
    try:
        for model in (actor, reference):
            for name, encoder in model.named_modules():
                if not isinstance(encoder, ImageEncoder):
                    continue
                if not encoder.frozen:
                    continue
                signature = encoder_signature(encoder)
                if signature is None:
                    continue
                saved.append(
                    (encoder, encoder._feature_cache, encoder._cache_signature)
                )
                other = active.get(name)
                encoder._feature_cache = (
                    other._feature_cache
                    if other is not None and _same_backbone(encoder, other)
                    else FrozenImageCache(budget)
                )
                encoder._cache_signature = signature
                active[name] = encoder
        with image_identity_scope(bool(saved)):
            yield
    finally:
        for encoder, cache, signature in reversed(saved):
            encoder._feature_cache = cache
            encoder._cache_signature = signature
