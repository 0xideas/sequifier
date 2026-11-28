"""Stable initialization namespaces, independent of runtime dropout."""

import hashlib
import json
import random
from contextlib import contextmanager

import numpy as np
import torch

from sequifier.contract_versions import ContractVersions


def derived_seed(seed: int | None, path: tuple[str, ...], phase: str) -> int | None:
    if seed is None:
        return None
    value = json.dumps(
        [ContractVersions.INITIALIZATION_SEED_DERIVATION, seed, path, phase],
        separators=(",", ":"),
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
    cuda_devices = []
    if torch.cuda.is_initialized():
        cuda_devices = (
            [torch.cuda.current_device()]
            if torch.distributed.is_initialized()
            else list(range(torch.cuda.device_count()))
        )
    cuda_states = {device: torch.cuda.get_rng_state(device) for device in cuda_devices}
    try:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.random.default_generator.manual_seed(seed)
        for device in cuda_states:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        for device, state in cuda_states.items():
            torch.cuda.set_rng_state(state, device)
