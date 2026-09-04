import math
from abc import ABC, abstractmethod
from typing import Annotated, Literal, Optional, Union

import torch
from pydantic import BaseModel, ConfigDict, Field

from sequifier.typechecking import beartype


class ProbabilityDistributionBaseClass(ABC):
    """
    Abstract base class for all probability distributions.
    """

    @abstractmethod
    @beartype
    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        pass


class GeometricDistribution(BaseModel, ProbabilityDistributionBaseClass):
    model_config = ConfigDict(extra="forbid")

    type: Literal["geometric"] = "geometric"
    p: float = Field(..., gt=0.0, le=1.0)

    @beartype
    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.p == 1.0:
            return torch.ones(shape, device=device, dtype=torch.long)

        uniform = torch.rand(shape, device=device, generator=generator)
        return (
            torch.floor(torch.log1p(-uniform) / math.log1p(-self.p)).to(torch.long) + 1
        )


class NormalDistributionDiscretizedFloor(BaseModel, ProbabilityDistributionBaseClass):
    model_config = ConfigDict(extra="forbid")

    type: Literal["normal_discretized_floor"] = "normal_discretized_floor"
    mean: float
    standard_deviation: float = Field(..., gt=0.0)

    @beartype
    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        val = (
            torch.randn(shape, device=device, generator=generator)
            * self.standard_deviation
            + self.mean
        )
        return torch.clamp(torch.round(val), min=0).long() + 1


class LogNormalDistributionDiscretizedFloor(
    BaseModel, ProbabilityDistributionBaseClass
):
    model_config = ConfigDict(extra="forbid")

    type: Literal["log_normal_discretized_floor",] = "log_normal_discretized_floor"
    mean: float
    standard_deviation: float = Field(..., gt=0.0)

    @beartype
    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        normal = (
            torch.randn(shape, device=device, generator=generator)
            * self.standard_deviation
            + self.mean
        )
        val = torch.exp(normal)
        return torch.round(val).long() + 1


class PoissonDistributionFloor(BaseModel, ProbabilityDistributionBaseClass):
    model_config = ConfigDict(extra="forbid")

    type: Literal["poisson_floor"] = "poisson_floor"
    rate: float = Field(..., gt=0.0)

    @beartype
    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        rate = torch.full(shape, self.rate, device=device)
        return torch.poisson(rate, generator=generator).long() + 1


ProbabilityDistribution = Annotated[
    Union[
        GeometricDistribution,
        NormalDistributionDiscretizedFloor,
        LogNormalDistributionDiscretizedFloor,
        PoissonDistributionFloor,
    ],
    Field(discriminator="type"),
]
