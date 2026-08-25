"""Canonical paths for model-scoped logging artifacts."""

from pathlib import Path

from sequifier.typechecking import beartype


@beartype
def model_log_directory(project_root: str | Path, model_name: str) -> Path:
    """Return the directory containing one model's logging artifacts."""
    return Path(project_root) / "logs" / model_name


@beartype
def rank_log_prefix(
    project_root: str | Path,
    model_name: str,
    rank: int,
    dataset_name: str | None = None,
    dataset_count: int = 1,
) -> Path:
    """Return the common filename prefix for one model rank's log files."""
    dataset = f"-{dataset_name}" if dataset_count > 1 and dataset_name else ""
    return model_log_directory(project_root, model_name) / (
        f"{model_name}{dataset}-rank{rank}"
    )


@beartype
def dataset_artifact_prefix(
    project_root: str | Path,
    model_name: str,
    *,
    dataset_name: str | None = None,
    dataset_count: int = 1,
) -> Path:
    """Return a dataset suffix only when more than one dataset is configured."""

    if dataset_count > 1 and dataset_name is None:
        raise ValueError("dataset_name is required for a multi-dataset artifact")
    suffix = f"-{dataset_name}" if dataset_count > 1 else ""
    return model_log_directory(project_root, model_name) / f"{model_name}{suffix}"


@beartype
def model_artifact_path(
    project_root: str | Path,
    model_name: str,
    artifact: str,
    extension: str,
    *,
    dataset_name: str | None = None,
    dataset_count: int = 1,
) -> Path:
    """Construct one canonical model or dataset-bound artifact path."""

    suffix = f"-{dataset_name}" if dataset_count > 1 and dataset_name else ""
    return (
        Path(project_root)
        / "models"
        / f"{model_name}{suffix}-{artifact}.{extension.lstrip('.')}"
    )
