import torch
import torch_optimizer  # noqa: F401

from sequifier.components import ComponentRegistry
from sequifier.optimizers.ademamix import AdEMAMix
from sequifier.typechecking import beartype

CUSTOM_OPTIMIZERS = {"AdEMAMix": AdEMAMix}
OPTIMIZER_REGISTRY = ComponentRegistry(
    {
        # Preserve the historical resolver precedence: custom, then
        # torch-optimizer, then PyTorch.
        **{
            name: value
            for name, value in vars(torch.optim).items()
            if isinstance(value, type) and issubclass(value, torch.optim.Optimizer)
        },
        **{
            name: value
            for name, value in vars(torch_optimizer).items()
            if isinstance(value, type) and issubclass(value, torch.optim.Optimizer)
        },
        **CUSTOM_OPTIMIZERS,
    },
    kind="optimizer",
)
SCHEDULER_REGISTRY = ComponentRegistry(
    {
        name: value
        for name, value in vars(torch.optim.lr_scheduler).items()
        if isinstance(value, type)
        and issubclass(
            value,
            (
                torch.optim.lr_scheduler.LRScheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ),
        )
    },
    kind="scheduler",
)


@beartype
def get_optimizer_class(optimizer_name: str) -> type[torch.optim.Optimizer]:
    """Resolve a custom, torch-optimizer, or torch optimizer class."""
    return OPTIMIZER_REGISTRY.resolve(optimizer_name)


@beartype
def get_scheduler_class(scheduler_name: str):
    """Resolve a supported PyTorch learning-rate scheduler."""
    return SCHEDULER_REGISTRY.resolve(scheduler_name)
