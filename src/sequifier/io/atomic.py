"""Atomic publication shared by model, checkpoint and trajectory writers."""

import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_path(path, *, durable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        yield temporary
        if durable:
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
        temporary.replace(path)
        if durable and os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save(path, payload, *, durable=True):
    import torch

    with atomic_path(path, durable=durable) as temporary:
        torch.save(payload, temporary)


def write_json_atomic(path, payload, **options):
    options = {"indent": 2, "allow_nan": False, **options}
    with atomic_path(path, durable=True) as temporary:
        temporary.write_text(json.dumps(payload, **options) + "\n", encoding="utf-8")
