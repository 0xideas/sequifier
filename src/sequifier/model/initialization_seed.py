"""Version 1 stable initialization namespaces, independent of runtime dropout."""

import hashlib
import json
import random
from contextlib import contextmanager

import numpy as np
import torch

SEED_DERIVATION_VERSION = 1


def derived_seed(seed: int | None, path: tuple[str, ...], phase: str) -> int | None:
    if seed is None:
        return None
    value = json.dumps(
        [SEED_DERIVATION_VERSION, seed, path, phase], separators=(",", ":")
    )
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % (2**63)


@contextmanager
def isolated_initialization(seed: int | None):
    if seed is None:
        yield
        return
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    )
    # Construction and initialization occur on CPU. Never initialize unrelated devices.
    try:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.random.default_generator.manual_seed(seed)
        if cuda_states is not None:
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
