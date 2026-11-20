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

    def validate_targets(self, targets, usage):
        targets = set(targets)
        if set(self.criterion) != targets:
            raise ValueError(f"{usage}: criterion keys must equal interface targets")
        for label, weights in (
            ("class_weights", self.class_weights),
            ("loss_weights", self.loss_weights),
        ):
            if weights is not None and weights.keys() - targets:
                raise ValueError(f"{usage}: {label} references unknown targets")
        if self.loss_weights is not None and all(
            self.loss_weights.get(target, 1.0) == 0 for target in targets
        ):
            raise ValueError(
                f"{usage}: at least one target must have positive loss weight"
            )

    def validate_interface(self, interface, usage):
        self.validate_targets(interface.target_columns, usage)
        for column in self.class_share_log_columns:
            if (
                interface.target_column_types.get(column) != "categorical"
                or column not in interface.id_maps
                or column not in interface.n_classes
            ):
                raise ValueError(
                    f"{usage}: class-share column {column!r} requires a categorical target with a codec"
                )
        for target, criterion in self.criterion.items():
            categorical = interface.target_column_types[target] == "categorical"
            if criterion == "AccurateCrossEntropyLoss" and not categorical:
                raise ValueError(
                    f"{usage}: AccurateCrossEntropyLoss requires categorical target {target!r}"
                )
            weights = (self.class_weights or {}).get(target)
            if target in interface.categorical_hashing:
                if weights is not None or criterion not in {
                    "CrossEntropyLoss",
                    "AccurateCrossEntropyLoss",
                }:
                    raise ValueError(
                        f"{usage}: hashed target {target!r} requires unweighted cross-entropy"
                    )
            if weights is not None and (
                not categorical
                or len(weights)
                not in {interface.n_classes[target], interface.target_n_classes[target]}
            ):
                raise ValueError(
                    f"{usage}: class_weights[{target!r}] must match the global or decoder classes of a categorical target"
                )

    @model_validator(mode="after")
    def losses(self):
        for name in self.criterion.values():
            if name != "AccurateCrossEntropyLoss" and name not in vars(torch.nn):
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
