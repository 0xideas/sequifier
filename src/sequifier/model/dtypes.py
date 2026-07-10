import torch
from torch import Tensor, nn


def module_param_dtype(module: nn.Module) -> torch.dtype | None:
    """Return the first floating parameter dtype for a module tree."""
    for parameter in module.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return None


def cast_floating_to_dtype(x: Tensor, dtype: torch.dtype) -> Tensor:
    if not x.is_floating_point() or x.dtype == dtype:
        return x
    return x.to(dtype=dtype)


def cast_floating_to_module_dtype(x: Tensor, module: nn.Module) -> Tensor:
    target_dtype = module_param_dtype(module)
    if target_dtype is None:
        return x
    return cast_floating_to_dtype(x, target_dtype)
