"""Validated seed windows and atomic numeric feature/mask advancement."""

from sequifier.io.values import (  # noqa: F401
    DomainValue,
    numeric_fields,
    numeric_scalar,
)
from sequifier.trajectory.schema import SeedHistory


def validate_seed(seed: SeedHistory) -> SeedHistory:
    return SeedHistory.model_validate(seed.model_dump())


def advance_history(
    history: SeedHistory, successor: dict[str, DomainValue]
) -> SeedHistory:
    return SeedHistory(
        observations={
            key: values[1:] + [successor[key]]
            for key, values in history.observations.items()
        },
        attention_valid_mask=history.attention_valid_mask[1:] + [True],
        static_inputs=dict(history.static_inputs),
    )
