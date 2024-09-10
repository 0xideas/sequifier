import torch
import torch_optimizer  # noqa: F401

from sequifier.optimizers.ademamix import AdEMAMix

CUSTOM_OPTIMIZERS = ["AdEMAMix"]

TORCH_OPTIMIZERS = [
    "A2GradUni",
    "A2GradInc",
    "A2GradExp",
    "AccSGD",
    "AdaBelief",
    "AdaBound",
    "Adafactor",
    "Adahessian",
    "AdaMod",
    "AdamP",
    "AggMo",
    "Apollo",
    "DiffGrad",
    "Lamb",
    "LARS",
    "Lion",
    "Lookahead",
    "MADGRAD",
    "NovoGrad",
    "PID",
    "QHAdam",
    "QHM",
    "RAdam",
    "SGDP",
    "SGDW",
    "Shampoo",
    "SWATS",
    "Yogi",
]


def get_optimizer_class(optimizer_name: str) -> torch.optim.Optimizer:
    if optimizer_name in CUSTOM_OPTIMIZERS:
        if optimizer_name == "AdEMAMix":
            return AdEMAMix
        else:
            raise Exception(f"Optimizer '{optimizer_name}' is not available")
    elif optimizer_name in TORCH_OPTIMIZERS:
        return eval(f"torch_optimizer.{optimizer_name}")
    else:
        return eval(f"torch.optim.{optimizer_name}")
