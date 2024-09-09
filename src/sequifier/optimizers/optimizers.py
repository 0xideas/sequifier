import torch

from sequifier.optimizers.ademamix import AdEMAMix

NON_STANDARD_OPTIMIZERS = ["AdEMAMix"]


def get_optimizer_class(optimizer_name: str) -> torch.optim.Optimizer:
    if optimizer_name in NON_STANDARD_OPTIMIZERS:
        if optimizer_name == "AdEMAMix":
            return AdEMAMix
        else:
            raise Exception(f"Optimizer '{optimizer_name}' is not available")
    else:
        return eval(f"torch.optim.{optimizer_name}")
