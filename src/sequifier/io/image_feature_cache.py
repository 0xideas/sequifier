"""Scoped, transient identities for owned policy frames; never checkpoint state."""

from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count

IMAGE_CACHE_PREFIX = "__image_cache_id:"
_enabled = ContextVar("sequifier_image_feature_cache", default=False)
_identities = count(1)


def image_feature_cache_enabled():
    return _enabled.get()


def new_image_identity():
    # Frames own their image data. A monotonic process-local identity avoids
    # hashing pixels or conflating visually identical but independently owned rows.
    return next(_identities)


@contextmanager
def image_identity_scope(enabled):
    token = _enabled.set(enabled)
    try:
        yield
    finally:
        _enabled.reset(token)
