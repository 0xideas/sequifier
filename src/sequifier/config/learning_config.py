"""Phase-owned supervised losses and sampling views."""

from typing import Annotated, Literal

from pydantic import Field, field_validator

from sequifier.config.components import BERTSpecModel, NextOccurrenceConfigModel
from sequifier.config.naming import ConfigModel

# The loss service passes either a vector of class logits and an integer label,
# or one scalar prediction and label, and requires one loss per valid token.
_CRITERION_TARGET_TYPES = {
    "AccurateCrossEntropyLoss": "categorical",
    "CrossEntropyLoss": "categorical",
    "MultiMarginLoss": "categorical",
    "NLLLoss": "categorical",
    "NLLLoss2d": "categorical",
    "BCELoss": "real",
    "BCEWithLogitsLoss": "real",
    "GaussianNLLLoss": "real",
    "HingeEmbeddingLoss": "real",
    "HuberLoss": "real",
    "KLDivLoss": "real",
    "L1Loss": "real",
    "MSELoss": "real",
    "PoissonNLLLoss": "real",
    "SmoothL1Loss": "real",
    "SoftMarginLoss": "real",
}


class TargetLoss(ConfigModel):
    criterion: str
    weight: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    class_weights: list[Annotated[float, Field(ge=0, allow_inf_nan=False)]] | None = (
        None
    )

    @field_validator("criterion")
    @classmethod
    def validate_criterion(cls, value):
        if value not in _CRITERION_TARGET_TYPES:
            raise ValueError(f"Unsupported supervised loss criterion {value!r}")
        return value


class DatasetLearningConfig(ConfigModel):
    losses: dict[str, TargetLoss] = Field(default_factory=dict)
    class_share_log_columns: list[str] = Field(default_factory=list)

    @property
    def criterion(self):
        return {target: loss.criterion for target, loss in self.losses.items()}

    @property
    def loss_weights(self):
        return {target: loss.weight for target, loss in self.losses.items()}

    @property
    def class_weights(self):
        return {
            target: loss.class_weights
            for target, loss in self.losses.items()
            if loss.class_weights is not None
        } or None

    def validate_targets(self, targets, usage):
        targets = set(targets)
        if set(self.losses) != targets:
            raise ValueError(
                f"{usage}: losses must cover exactly the interface targets"
            )
        if not any(loss.weight > 0 for loss in self.losses.values()):
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
        for target, loss in self.losses.items():
            criterion = loss.criterion
            categorical = interface.target_column_types[target] == "categorical"
            weights = loss.class_weights
            if target in interface.categorical_hashing:
                if weights is not None or criterion not in {
                    "CrossEntropyLoss",
                    "AccurateCrossEntropyLoss",
                }:
                    raise ValueError(
                        f"{usage}: hashed target {target!r} requires unweighted cross-entropy"
                    )
            required = _CRITERION_TARGET_TYPES[criterion]
            if required != interface.target_column_types[target]:
                raise ValueError(
                    f"{usage}: {criterion} requires a {required} target for {target!r}"
                )
            if weights is not None and (
                not categorical
                or len(weights)
                not in {interface.n_classes[target], interface.target_n_classes[target]}
            ):
                raise ValueError(
                    f"{usage}: class_weights[{target!r}] must match the global or decoder classes of a categorical target"
                )


class PredictionObjective(ConfigModel):
    type: Literal["causal", "policy", "final_value"]


class MaskedObjective(BERTSpecModel):
    type: Literal["bert"]


class EventObjective(NextOccurrenceConfigModel):
    type: Literal["next_occurrence"]


class SupervisedLearningConfig(ConfigModel):
    datasets: dict[str, DatasetLearningConfig] = Field(default_factory=dict)
    objective: (
        Annotated[
            PredictionObjective | MaskedObjective | EventObjective,
            Field(discriminator="type"),
        ]
        | None
    ) = None
    window_stride: int | None = Field(default=None, gt=0)

    def objective_name(self, prediction_view: str) -> str:
        if self.objective is not None:
            return self.objective.type
        return (
            "causal"
            if prediction_view in {"causal", "final_value", "next_occurrence"}
            else prediction_view
        )

    @property
    def bert_spec(self):
        return self.objective if isinstance(self.objective, MaskedObjective) else None

    @property
    def next_occurrence_config(self):
        return self.objective if isinstance(self.objective, EventObjective) else None
