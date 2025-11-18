import torch
import torch_optimizer  # noqa: F401

from sequifier.optimizers.ademamix import AdEMAMix

CUSTOM_OPTIMIZERS = {"AdEMAMix": AdEMAMix}


def get_optimizer_class(optimizer_name: str) -> torch.optim.Optimizer:
    """Gets the optimizer class from a string.

    Args:
        optimizer_name: The name of the optimizer.

    Returns:
        The optimizer class.
    """
    if optimizer_name in CUSTOM_OPTIMIZERS:
        return CUSTOM_OPTIMIZERS[optimizer_name]
    elif hasattr(torch_optimizer, optimizer_name):
        return getattr(torch_optimizer, optimizer_name)
    elif hasattr(torch.optim, optimizer_name):
        return getattr(torch.optim, optimizer_name)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not found.")
