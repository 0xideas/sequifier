from pathlib import Path
from typing import Any


def checkpoint_path(training_config: Any) -> Path:
    resume = training_config.training_spec.resume
    configured_path = resume.checkpoint_path if resume is not None else None
    if configured_path:
        path = Path(configured_path)
        if not path.is_absolute():
            path = Path(training_config.project_root) / path
    else:
        path = (
            Path(training_config.project_root)
            / "checkpoints"
            / "runs"
            / training_config.model_name
            / f"{training_config.model_name}-latest.pt"
        )
    return path.resolve()


def select_run_checkpoint(training_config: Any) -> dict[str, Any] | None:
    resume = training_config.training_spec.resume
    if resume is None or resume.policy == "never":
        return None
    path = checkpoint_path(training_config)
    if path.is_file():
        return {"path": str(path)}
    if resume.policy == "required":
        raise FileNotFoundError(f"Required run checkpoint does not exist: {path}")
    return None
