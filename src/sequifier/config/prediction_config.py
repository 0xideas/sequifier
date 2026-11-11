"""Model-owned prediction parameters, independent of task bindings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredictionConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ModelScaleConfig(PredictionConfigBase):
    """Model-owned Gaussian scale, including optional learned parameters."""

    initial: float = Field(gt=0)
    trainable: bool = False
    min: float = Field(default=1e-4, gt=0)
    max: float = Field(default=100.0, gt=0)

    @model_validator(mode="after")
    def validate_range(self):
        if not self.min <= self.initial <= self.max:
            raise ValueError("scale.initial must lie between scale.min and scale.max")
        return self


class PredictionDistributionConfig(PredictionConfigBase):
    """A model output's Gaussian parameterization, independent of policy behavior."""

    type: Literal["real"] = "real"
    scale: ModelScaleConfig
