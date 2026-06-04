from abc import ABC, abstractmethod
from typing import Annotated, Literal, Union

import torch
from pydantic import BaseModel, Field


class ProbabilityDistributionBaseClass(ABC):
    """
    Abstract base class for all probability distributions.
    """

    @abstractmethod
    def sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        pass


class GeometricDistribution(BaseModel, ProbabilityDistributionBaseClass):
    type: Literal["GeometricDistribution"] = "GeometricDistribution"
    p: float = Field(..., gt=0.0, le=1.0)

    def sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        if self.p == 1.0:
            return torch.ones(shape, device=device, dtype=torch.long)

        # torch.distributions.Geometric models the number of failures before the first success.
        # Adding 1 converts it to the total number of trials (span length).
        m = torch.distributions.Geometric(probs=torch.tensor([self.p], device=device))
        return (m.sample(shape).squeeze(-1) + 1).long()


class NormalDistributionDiscretizedFloor(BaseModel, ProbabilityDistributionBaseClass):
    type: Literal["NormalDistributionDiscretizedFloor"] = (
        "NormalDistributionDiscretizedFloor"
    )
    mean: float
    standard_deviation: float = Field(..., gt=0.0)

    def sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        val = torch.normal(
            mean=self.mean, std=self.standard_deviation, size=shape, device=device
        )
        return torch.clamp(torch.round(val), min=0).long() + 1


class LogNormalDistributionDistcretizedFloor(
    BaseModel, ProbabilityDistributionBaseClass
):
    type: Literal["LogNormalDistributionDistcretizedFloor"] = (
        "LogNormalDistributionDistcretizedFloor"
    )
    mean: float
    standard_deviation: float = Field(..., gt=0.0)

    def sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        m = torch.distributions.LogNormal(
            loc=torch.tensor([self.mean], device=device),
            scale=torch.tensor([self.standard_deviation], device=device),
        )
        val = m.sample(shape).squeeze(-1)
        return torch.round(val).long() + 1


class PoissonDistributionFloor(BaseModel, ProbabilityDistributionBaseClass):
    type: Literal["PoissonDistributionFloor"] = "PoissonDistributionFloor"
    rate: float = Field(..., gt=0.0)

    def sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        m = torch.distributions.Poisson(rate=torch.tensor([self.rate], device=device))
        return m.sample(shape).squeeze(-1).long()


ProbabilityDistribution = Annotated[
    Union[
        GeometricDistribution,
        NormalDistributionDiscretizedFloor,
        LogNormalDistributionDistcretizedFloor,
        PoissonDistributionFloor,
    ],
    Field(discriminator="type"),
]
