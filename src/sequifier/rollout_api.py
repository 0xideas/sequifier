"""Public API for local numeric forecast adapters and trajectory collection."""

from sequifier.config.rollout_config import AdapterConfig, InputBinding, RolloutConfig
from sequifier.rollout.contracts import (
    EnvironmentAdapter,
    ResetRequest,
    StepRequest,
    StepResult,
)
from sequifier.rollout.coordinator import collect_rollout
from sequifier.trajectory import (
    Trajectory,
    load_trajectory,
    save_trajectory,
    validate_trajectory,
)
from sequifier.trajectory.schema import SeedHistory

__all__ = [
    "AdapterConfig",
    "InputBinding",
    "RolloutConfig",
    "EnvironmentAdapter",
    "ResetRequest",
    "StepRequest",
    "StepResult",
    "SeedHistory",
    "Trajectory",
    "collect_rollout",
    "load_trajectory",
    "save_trajectory",
    "validate_trajectory",
]
