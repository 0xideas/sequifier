"""Composition-root services for Sequifier training runs.

Keep package exports lazy so importing a leaf service such as ``random_state``
does not load the run builder and recursively import its export services.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sequifier.runtime.builder import RunBuilder, TrainingRun
    from sequifier.runtime.context import ExecutionEnvironment, RunContext
    from sequifier.runtime.random_state import RandomStateManager

__all__ = [
    "ExecutionEnvironment",
    "RandomStateManager",
    "RunBuilder",
    "RunContext",
    "TrainingRun",
]


def __getattr__(name: str) -> Any:
    if name in {"RunBuilder", "TrainingRun"}:
        from sequifier.runtime.builder import RunBuilder, TrainingRun

        return {"RunBuilder": RunBuilder, "TrainingRun": TrainingRun}[name]
    if name in {"ExecutionEnvironment", "RunContext"}:
        from sequifier.runtime.context import ExecutionEnvironment, RunContext

        return {
            "ExecutionEnvironment": ExecutionEnvironment,
            "RunContext": RunContext,
        }[name]
    if name == "RandomStateManager":
        from sequifier.runtime.random_state import RandomStateManager

        return RandomStateManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
