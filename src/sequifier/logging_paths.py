"""Canonical paths for model-scoped logging artifacts."""

from pathlib import Path


def model_log_directory(project_root: str | Path, model_name: str) -> Path:
    """Return the directory containing one model's logging artifacts."""
    return Path(project_root) / "logs" / model_name


def rank_log_prefix(
    project_root: str | Path,
    model_name: str,
    rank: int,
) -> Path:
    """Return the common filename prefix for one model rank's log files."""
    return model_log_directory(project_root, model_name) / (
        f"sequifier-{model_name}-rank{rank}"
    )
