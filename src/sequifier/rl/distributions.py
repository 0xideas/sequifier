"""The categorical action distribution used by learning, scoring, and execution."""

import torch
from torch import Tensor


def action_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(
        device=device if device.type == "cuda" else "cpu"
    ).manual_seed(seed)


class CategoricalActionDistribution:
    def __init__(self, logits: Tensor):
        if logits.ndim < 2 or logits.shape[-1] < 2 or not torch.isfinite(logits).all():
            raise ValueError(
                "categorical policy logits must be finite with at least two actions"
            )
        self.logits = logits
        self.distribution = torch.distributions.Categorical(logits=logits)

    def mode(self) -> Tensor:
        return self.logits.argmax(dim=-1)

    def select(self, deterministic: bool, generator: torch.Generator) -> Tensor:
        return self.mode() if deterministic else self.sample(generator)

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        probabilities = self.distribution.probs
        # MPS has no device-specific Generator; sample on its generator's device.
        sampling_device = (
            generator.device if generator is not None else probabilities.device
        )
        sampled = torch.multinomial(
            probabilities.to(sampling_device).reshape(-1, probabilities.shape[-1]),
            1,
            generator=generator,
        ).to(probabilities.device)
        return sampled.reshape(probabilities.shape[:-1])

    def log_prob(self, actions: Tensor) -> Tensor:
        if actions.dtype != torch.int64 or actions.shape != self.logits.shape[:-1]:
            raise ValueError(
                "categorical action IDs must be int64 and match the logits batch shape"
            )
        if (actions < 0).any() or (actions >= self.logits.shape[-1]).any():
            raise ValueError("categorical action ID is out of range")
        return self.distribution.log_prob(actions)

    def entropy(self) -> Tensor:
        return self.distribution.entropy()


def masked_action_nll_loss(
    distribution: CategoricalActionDistribution, actions: Tensor, valid_mask: Tensor
) -> Tensor:
    if (
        valid_mask.dtype != torch.bool
        or valid_mask.shape != actions.shape
        or actions.shape != distribution.logits.shape[:-1]
    ):
        raise ValueError("action loss masks must be boolean and match actions/logits")
    if not valid_mask.any():
        raise ValueError("action likelihood requires at least one labeled decision")
    selected = CategoricalActionDistribution(distribution.logits[valid_mask])
    return -selected.log_prob(actions[valid_mask]).mean()
