"""Shared path geometry and destination checks for training outputs."""

from pathlib import Path


def unique_paths(paths, message):
    paths = [Path(path).resolve() for path in paths]
    if len(set(paths)) != len(paths):
        raise ValueError(message)
    return paths


def contains_path(directory, path):
    return directory == path or directory in path.parents


def paths_overlap(left, right):
    return contains_path(left, right) or contains_path(right, left)


def validate_path_conflicts(paths, protected, *, label, check_each_other=True):
    protected = {Path(path).resolve() for path in protected}
    paths = [Path(path).resolve() for path in paths]
    for path in paths:
        others = protected | (set(paths) - {path} if check_each_other else set())
        for other in others:
            if paths_overlap(path, other):
                raise ValueError(f"{label} {path} conflicts with input/output {other}")


def require_file_destinations(paths):
    for path in paths:
        if path.is_dir() or any(p.exists() and not p.is_dir() for p in path.parents):
            raise ValueError(f"output requires a file destination: {path}")


def require_directory_destinations(paths):
    for path in paths:
        if path.exists() and not path.is_dir():
            raise ValueError(f"archive requires a directory: {path}")


def validate_training_destinations(config, checkpoint, protected):
    """Reserve generated trees as well as explicit files before any publication."""
    checkpoint = Path(checkpoint).resolve()
    files = unique_paths(
        [config.model_path, config.report_path, checkpoint],
        "training model, report and checkpoint paths must be distinct",
    )
    manifest = checkpoint.parent / "manifest.json"
    if manifest != Path(config.report_path).resolve():
        files = unique_paths([*files, manifest], "manifest overlaps a training output")
    directories = [Path(config.phase_directory).resolve()]
    from sequifier.logging_paths import model_log_directory

    directories.append(
        model_log_directory(config.project_root, config.model_name).resolve()
    )
    phases = getattr(config, "phases", getattr(config, "training_plan", ()))
    from sequifier.artifacts.phase import PhaseArtifacts

    generated = []
    for index, phase in enumerate(phases):
        artifacts = PhaseArtifacts.for_phase(config.phase_directory, index, phase.name)
        for path in (
            artifacts.model,
            artifacts.report,
            artifacts.model.parent / "commits",
            artifacts.model.with_name("run.pt"),
            artifacts.model.parent / "native",
            artifacts.model.parent / "checkpoints",
            artifacts.telemetry.parent,
            artifacts.telemetry,
        ):
            resolved = path.resolve()
            if not contains_path(directories[0], resolved):
                raise ValueError(f"phase destination escapes phase_directory: {path}")
            generated.append(resolved)
    validate_path_conflicts(
        generated, protected, label="phase output", check_each_other=False
    )
    directories.extend(
        Path(phase.archive_directory).resolve()
        for phase in phases
        if getattr(phase, "archive_directory", None)
    )
    directories = unique_paths(
        directories, "training output directories must be distinct"
    )
    require_file_destinations(files)
    require_directory_destinations(directories)
    # A generated snapshot may be selected as the rolling checkpoint on resume.
    snapshot_dirs = {
        (
            PhaseArtifacts.for_phase(
                config.phase_directory, index, phase.name
            ).model.parent
            / "checkpoints"
        ).resolve()
        for index, phase in enumerate(phases)
    }
    resume = getattr(getattr(config, "runtime_settings", None), "resume", None)
    snapshot_resume = (
        resume is not None
        and resume.policy != "never"
        and checkpoint.is_file()
        and checkpoint.parent in snapshot_dirs
        and checkpoint.name.startswith(f"{config.model_name}-")
        and checkpoint.suffix == ".pt"
        and manifest != Path(config.report_path).resolve()
    )
    if snapshot_resume:
        external_files = [path for path in files if path not in {checkpoint, manifest}]
        validate_path_conflicts(
            [*external_files, *directories], protected, label="training output"
        )
        validate_path_conflicts(
            [checkpoint, manifest],
            protected | set(external_files) | set(directories[1:]),
            label="training output",
        )
    else:
        validate_path_conflicts(
            [*files, *directories], protected, label="training output"
        )
    if any(phase.algorithm == "supervised" for phase in phases):
        import re

        export_directory = (Path(config.project_root) / "models").resolve()
        names = [config.model_name] + [
            f"{config.model_name}-{name}" for name in config.dataset_training
        ]
        pattern = re.compile(
            "(?:"
            + "|".join(re.escape(name) for name in names)
            + r")-(?:best|last)(?:-embedding)?-\d+\.(?:pt|onnx)"
        )
        for path in [*files, *directories, *(Path(p).resolve() for p in protected)]:
            if contains_path(path, export_directory) or (
                path.parent == export_directory and pattern.fullmatch(path.name)
            ):
                raise ValueError(
                    f"training destination/input overlaps generated exports: {path}"
                )


def resolve_training_output_paths(config):
    root = Path(config.project_root)
    defaults = {
        "model_path": f"models/{config.model_name}.pt",
        "report_path": f"outputs/training/{config.model_name}/report.json",
        "phase_directory": f"outputs/training/{config.model_name}/phases",
    }
    for name, default in defaults.items():
        setattr(
            config, name, str((root / (getattr(config, name, "") or default)).resolve())
        )
