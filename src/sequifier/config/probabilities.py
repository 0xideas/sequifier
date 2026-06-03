import math
import random
from abc import ABC, abstractmethod
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

epsilon = 1e-20


class ProbabilityDistributionBaseClass(ABC):
    """
    Abstract base class for all probability distributions.
    """

    @abstractmethod
    def sample(self) -> Union[int, float]:
        pass


class GeometricDistribution(BaseModel, ProbabilityDistributionBaseClass):
    type: Literal["GeometricDistribution"] = "GeometricDistribution"
    p: float = Field(..., gt=0.0, le=1.0)

    def sample(self) -> int:
        if self.p == 1.0:
            return 1
        return math.ceil(math.log(random.random() + epsilon) / math.log(1.0 - self.p))


class NormalDistributionDiscretizedFloor(BaseModel, ProbabilityDistributionBaseClass):
    type: Literal["NormalDistributionDiscretizedFloor"] = (
        "NormalDistributionDiscretizedFloor"
    )
    mean: float
    standard_deviation: float = Field(..., gt=0.0)

    def sample(self) -> int:
        val = random.gauss(self.mean, self.standard_deviation)
        return max(round(val), 0) + 1


class LogNormalDistributionDistcretizedFloor(
    BaseModel, ProbabilityDistributionBaseClass
):
    type: Literal["LogNormalDistributionDistcretizedFloor"] = (
        "LogNormalDistributionDistcretizedFloor"
    )
    mean: float
    standard_deviation: float = Field(..., gt=0.0)

    def sample(self) -> int:
        val = random.lognormvariate(self.mean, self.standard_deviation)
        return round(val) + 1


class PoissonDistributionFloor(BaseModel, ProbabilityDistributionBaseClass):
    type: Literal["PoissonDistributionFloor"] = "PoissonDistributionFloor"
    rate: float = Field(..., gt=0.0)

    def sample(self) -> int:
        L = math.exp(-self.rate)
        k = 0
        p = 1.0

        while p > L:
            k += 1
            p *= random.random()

        return k


ProbabilityDistribution = Annotated[
    Union[
        GeometricDistribution,
        NormalDistributionDiscretizedFloor,
        LogNormalDistributionDistcretizedFloor,
        PoissonDistributionFloor,
    ],
    Field(discriminator="type"),
]
