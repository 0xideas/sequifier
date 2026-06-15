import torch
import torch_optimizer  # noqa: F401

from sequifier.optimizers.ademamix import AdEMAMix

CUSTOM_OPTIMIZERS = {"AdEMAMix": AdEMAMix}


def get_optimizer_class(optimizer_name: str) -> torch.optim.Optimizer:
    """Resolve a custom, torch-optimizer, or torch optimizer class."""
    if optimizer_name in CUSTOM_OPTIMIZERS:
        return CUSTOM_OPTIMIZERS[optimizer_name]
    elif hasattr(torch_optimizer, optimizer_name):
        return getattr(torch_optimizer, optimizer_name)
    elif hasattr(torch.optim, optimizer_name):
        return getattr(torch.optim, optimizer_name)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not found.")
