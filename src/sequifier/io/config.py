"""Small compatibility boundary for standalone data-loader tests and callers."""

from typing import Any

from sequifier.typechecking import beartype


@beartype
def global_training(config: Any) -> Any:
    training = vars(config).get("global_training")
    if training is not None:
        return training
    return config.training_spec
