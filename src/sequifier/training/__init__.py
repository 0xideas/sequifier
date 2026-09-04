"""Lifecycle-aware training orchestration and runtime services.

The package initializer stays lazy so importing a leaf service such as
``sequifier.training.loss`` cannot recursively import the training engine through
the evaluation package.
"""

from typing import Any

__all__ = ["OptimizationRuntime", "RunResult", "RunState", "TrainingEngine"]


def __getattr__(name: str) -> Any:
    if name in {"RunResult", "TrainingEngine"}:
        from sequifier.training.engine import RunResult, TrainingEngine

        return {"RunResult": RunResult, "TrainingEngine": TrainingEngine}[name]
    if name == "OptimizationRuntime":
        from sequifier.training.optimization import OptimizationRuntime

        return OptimizationRuntime
    if name == "RunState":
        from sequifier.training.state import RunState

        return RunState
    raise AttributeError(name)
