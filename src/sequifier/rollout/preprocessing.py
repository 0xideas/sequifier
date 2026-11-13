"""Fixed numeric training transforms; never fit on seeds or adapter values."""

import numpy as np

from sequifier.io.preprocessing import (
    NORMALIZATION_EPSILON,
    denormalize_values,
    normalization_statistics,
    normalize_values,
)
from sequifier.rollout.history import validate_seed
from sequifier.trajectory.schema import SeedHistory


class FixedNumericPreprocessing:
    def __init__(self, columns, statistics, normalize, dtypes):
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
            values = normalize_values(values, self.statistics[column])
        with np.errstate(over="ignore", invalid="ignore"):
            values = values.astype(self.dtypes[column])
        if not np.isfinite(values).all():
            raise ValueError(
                f"rollout field {column!r} is incompatible with its model dtype/scale"
            )
        return values

    def seed_from_model(self, inputs, mask, input_columns, static_columns):
        if mask.ndim != 2 or mask.shape[0] != 1 or mask.dtype != np.bool_:
            raise ValueError(
                "rollout seed requires a boolean attention mask for one window"
            )
        observations, static = {}, {}
        for column in input_columns:
            values = inputs[column]
            if values.shape != mask.shape:
                raise ValueError(
                    f"rollout seed {column!r} must be a scalar feature window matching its mask"
                )
            # Padding is represented by neutral domain values, never decoded as observations.
            decoded = np.zeros(values.shape, dtype=np.float64)
            decoded[mask] = self.decode(column, values[mask])
            observations[column] = decoded[0].tolist()
        for column in static_columns:
            if inputs[column].shape != (1,):
                raise ValueError(
                    f"rollout static input {column!r} must have shape (1,)"
                )
            static[column] = float(self.decode(column, inputs[column])[0])
        return validate_seed(
            SeedHistory(
                observations=observations,
                attention_valid_mask=mask[0].tolist(),
                static_inputs=static,
            )
        )

    def model_inputs(self, history: SeedHistory):
        mask = np.asarray([history.attention_valid_mask], dtype=np.bool_)
        features = {}
        for column, values in history.observations.items():
            encoded = np.zeros(mask.shape, dtype=self.dtypes[column])
            # False-mask padding is not a physical value to normalize.
            encoded[mask] = self.encode(column, np.asarray([values])[mask])
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
