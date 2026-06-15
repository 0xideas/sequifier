import pytest
import torch

from sequifier.config.probabilities import (
    GeometricDistribution,
    LogNormalDistributionDiscretizedFloor,
    NormalDistributionDiscretizedFloor,
    PoissonDistributionFloor,
)


@pytest.mark.parametrize(
    "distribution",
    [
        GeometricDistribution(p=0.35),
        NormalDistributionDiscretizedFloor(mean=2.0, standard_deviation=0.5),
        LogNormalDistributionDiscretizedFloor(mean=0.5, standard_deviation=0.4),
        PoissonDistributionFloor(rate=2.0),
    ],
)
def test_probability_distributions_use_local_generator_without_global_rng(
    distribution,
):
    device = torch.device("cpu")
    torch.manual_seed(123)
    rng_state = torch.get_rng_state().clone()

    first_generator = torch.Generator(device=device)
    first_generator.manual_seed(99)
    first_samples = distribution.sample(
        (64,),
        device=device,
        generator=first_generator,
    )
    after_first = torch.get_rng_state().clone()

    second_generator = torch.Generator(device=device)
    second_generator.manual_seed(99)
    second_samples = distribution.sample(
        (64,),
        device=device,
        generator=second_generator,
    )
    after_second = torch.get_rng_state().clone()

    assert torch.equal(first_samples, second_samples)
    assert first_samples.device == device
    assert first_samples.dtype == torch.long
    assert torch.all(first_samples >= 1)
    assert torch.equal(after_first, rng_state)
    assert torch.equal(after_second, rng_state)
