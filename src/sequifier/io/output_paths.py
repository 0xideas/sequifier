"""Shared path geometry and destination checks."""

import re
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


def validate_inference_destination(destination, inputs, *, label):
    entry = Path(destination)
    # Atomic replacement acts on the directory entry even if it is a symlink.
    entry = entry.parent.resolve() / entry.name
    resolved = entry.resolve()
    for input_path in inputs:
        source = Path(input_path).resolve()
        conflict = paths_overlap(resolved, source) or paths_overlap(entry, source)
        if not conflict:
            try:
                conflict = resolved.samefile(source)
            except (FileNotFoundError, NotADirectoryError):
                pass
        if conflict:
            raise ValueError(
                f"{label} {destination!s} conflicts with input {input_path}"
            )


def require_file_destinations(paths):
    for path in paths:
        if path.is_dir() or any(p.exists() and not p.is_dir() for p in path.parents):
            raise ValueError(f"output requires a file destination: {path}")


def require_directory_destinations(paths):
    for path in paths:
        if path.exists() and not path.is_dir():
            raise ValueError(f"archive requires a directory: {path}")


def validate_training_destinations(config, checkpoint, protected, *, phase_offset=0):
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
    from sequifier.rl.run_config import ResolvedPolicyRun

    if isinstance(config, ResolvedPolicyRun):
        phases, resume = config.phases, config.resume
        dataset_names = ()
    else:
        phases, resume = config.training_plan, config.runtime_settings.resume
        dataset_names = tuple(config.dataset_training)
    from sequifier.artifacts.phase import PhaseArtifacts

    generated = []
    for index, phase in enumerate(phases, phase_offset):
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
        Path(archive).resolve()
        for phase in phases
        if (archive := phase.rollout_archive_directory)
    )
    directories = unique_paths(
        directories, "training output directories must be distinct"
    )
    require_file_destinations(files)
    require_directory_destinations(directories)
    # A generated snapshot may be selected as the rolling checkpoint on resume.
    snapshot_phases = {
        (
            PhaseArtifacts.for_phase(
                config.phase_directory, index, phase.name
            ).model.parent
            / "checkpoints"
        ).resolve(): phase
        for index, phase in enumerate(phases, phase_offset)
    }
    snapshot_phase = snapshot_phases.get(checkpoint.parent)
    snapshot_name = snapshot_phase is not None and (
        checkpoint.name.startswith(f"{config.model_name}-")
        if snapshot_phase.algorithm == "supervised"
        else re.fullmatch(r"transaction-\d{8,}\.pt", checkpoint.name) is not None
    )
    snapshot_resume = (
        resume is not None
        and resume.policy != "never"
        and checkpoint.is_file()
        and snapshot_name
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
        export_directory = (Path(config.project_root) / "models").resolve()
        names = [config.model_name] + [
            f"{config.model_name}-{name}" for name in dataset_names
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
    output_dir = Path("outputs/training") / config.model_name
    config.model_path = str(
        (root / (config.model_path or f"models/{config.model_name}.pt")).resolve()
    )
    config.report_path = str(
        (root / (config.report_path or output_dir / "report.json")).resolve()
    )
    config.phase_directory = str(
        (root / (config.phase_directory or output_dir / "phases")).resolve()
    )
