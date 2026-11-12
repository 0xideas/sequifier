"""Phase-owned supervised losses and sampling views."""

import math
from typing import Optional

import torch
from pydantic import ConfigDict, Field, model_validator

from sequifier.config.components import BERTSpecModel, NextOccurrenceConfigModel
from sequifier.config.naming import ConfigModel


class DatasetLearningConfig(ConfigModel):
    model_config = ConfigDict(extra="forbid")
    criterion: dict[str, str] = Field(default_factory=dict)
    class_weights: Optional[dict[str, list[float]]] = None
    loss_weights: Optional[dict[str, float]] = None
    class_share_log_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def losses(self):
        for name in self.criterion.values():
            if name != "AccurateCrossEntropyLoss" and not hasattr(torch.nn, name):
                raise ValueError(f"unknown criterion {name!r}")
        for weights in (self.loss_weights, self.class_weights):
            for value in (weights or {}).values():
                values = value if isinstance(value, list) else [value]
                if any(not math.isfinite(x) or x < 0 for x in values):
                    raise ValueError(
                        "loss and class weights must be finite and nonnegative"
                    )
        return self


class SupervisedLearningConfig(ConfigModel):
    model_config = ConfigDict(extra="forbid")
    datasets: dict[str, DatasetLearningConfig] = Field(default_factory=dict)
    bert_spec: BERTSpecModel | None = None
    next_occurrence_config: NextOccurrenceConfigModel | None = None
    window_stride: int | None = Field(default=None, gt=0)
