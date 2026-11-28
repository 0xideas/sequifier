"""Fixed numeric training transforms; never fit on seeds or adapter values."""

import numpy as np

from sequifier.config.column_names import INCOMING_ACTION_PRESENT
from sequifier.io.preprocessing import (
    NORMALIZATION_EPSILON,
    denormalize_values,
    normalization_statistics,
    normalize_input_values,
)
from sequifier.trajectory.schema import SeedHistory


class FixedNumericPreprocessing:
    def __init__(self, columns, statistics, normalize, dtypes, incoming_columns=()):
        self.feedback_columns = {*incoming_columns, INCOMING_ACTION_PRESENT}
        self.normalize = normalize
        self.statistics = (
            normalization_statistics(
                {column: statistics.get(column, {}) for column in columns}
            )
            if normalize
            else {}
        )
        self.dtypes = {column: np.dtype(dtypes[column]) for column in columns}
        for column in columns:
            if self.dtypes[column].kind != "f":
                raise ValueError(
                    f"rollout field {column!r} must have a floating numeric dtype"
                )

    def decode(self, column, values):
        values = np.asarray(values, dtype=np.float64)
        if self.normalize:
            values = denormalize_values(values, self.statistics[column])
        if not np.isfinite(values).all():
            raise ValueError(f"rollout field {column!r} decoded to non-finite values")
        return values

    def encode(self, column, values):
        values = np.asarray(values, dtype=np.float64)
        if self.normalize:
            values = normalize_input_values(
                values, self.statistics[column], self.dtypes[column].name
            )
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray(values, dtype=self.dtypes[column])
        if not np.isfinite(values).all():
            raise ValueError(
                f"rollout field {column!r} is incompatible with its model dtype/scale"
            )
        return values

    def model_inputs(self, history: SeedHistory):
        mask = np.asarray([history.attention_valid_mask], dtype=np.bool_)
        visible = mask & ~np.asarray([history.observation_mask], dtype=np.bool_)
        features = {}
        for column, values in history.observations.items():
            encoded = np.zeros(mask.shape, dtype=self.dtypes[column])
            # False-mask padding is not a physical value to normalize.
            selected = mask if column in self.feedback_columns else visible
            encoded[selected] = self.encode(column, np.asarray([values])[selected])
            features[column] = encoded
        features.update(
            {
                column: self.encode(column, [value])
                for column, value in history.static_inputs.items()
            }
        )
        return features, {"attention_valid_mask": mask}

    def to_dict(self):
        return {
            "normalize_real_columns": self.normalize,
            "selected_columns_statistics": self.statistics,
            "model_dtypes": {
                column: dtype.name for column, dtype in self.dtypes.items()
            },
            "epsilon": NORMALIZATION_EPSILON,
        }
