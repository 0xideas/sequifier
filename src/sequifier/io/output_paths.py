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
    for index, path in enumerate(paths):
        # Skip this entry by position so equal destinations still conflict.
        others = protected | (set(paths[index + 1 :]) if check_each_other else set())
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
            raise ValueError(f"output requires a directory: {path}")


def run_manifest_path(config):
    return (
        Path(config.project_root) / "checkpoints" / f"{config.model_name}.json"
    ).resolve()


def validate_training_destinations(
    config, checkpoint, protected, *, phase_offset=0, group=False
):
    """Reserve generated trees as well as explicit files before any publication."""
    checkpoint = Path(checkpoint).resolve()
    phase_root = Path(config.outputs.phase_directory).resolve()
    directories = []
    files = [checkpoint]
    manifest = run_manifest_path(config)
    if not group:
        directories.append(phase_root)
        files.extend([config.outputs.model_path, config.outputs.report_path])
        if manifest != Path(config.outputs.report_path).resolve():
            files.append(manifest)
    from sequifier.logging_paths import model_log_directory

    directories.append(
        model_log_directory(config.project_root, config.model_name).resolve()
    )
    from sequifier.rl.run_config import ResolvedPolicyRun

    if isinstance(config, ResolvedPolicyRun):
        phases, resume = config.phases, config.resume
        dataset_names = ()
    else:
        phases, resume = config.training_plan, config.run.recovery.resume
        dataset_names = tuple(config.dataset_training)
    from sequifier.artifacts.phase import PhaseArtifacts

    root = Path(config.project_root)
    deferred = getattr(config, "_rollout_plan", None)
    if deferred is not None and not group:
        phases = deferred.ordered_phases
        protected = set(protected)
        for dataset in deferred.datasets.values():
            for source in dataset.sources.values():
                for field in (
                    "metadata_config_path",
                    "manifest_path",
                    "preprocessing_config_path",
                    "static_data_path",
                    "path",
                ):
                    if value := getattr(source, field, None):
                        protected.add((root / value).resolve())
                protected.update(
                    (root / path).resolve() for path in getattr(source, "paths", ())
                )
        for phase in phases:
            reference = getattr(phase, "reference", None)
            if path := getattr(reference, "path", None):
                protected.add((root / path).resolve())

    sequence_manifests = []
    phase_files = []
    snapshot_phases = {}
    generated = []
    for index, phase in enumerate(phases, phase_offset):
        artifacts = PhaseArtifacts.for_phase(
            config.outputs.phase_directory, index, phase.name
        )
        if group:
            directories.append(artifacts.model.parent.resolve())
        snapshot_phases[artifacts.snapshots.resolve()] = phase
        phase_files.extend([artifacts.model, artifacts.report])
        for path in (
            artifacts.model,
            artifacts.report,
            artifacts.commits,
            artifacts.recovery,
            artifacts.snapshots,
            artifacts.telemetry.parent,
            artifacts.telemetry,
        ):
            resolved = path.resolve()
            if not contains_path(phase_root, resolved):
                raise ValueError(f"phase destination escapes phase_directory: {path}")
            generated.append(resolved)
    validate_path_conflicts(
        generated, protected, label="phase output", check_each_other=False
    )
    for phase in phases:
        output = getattr(getattr(phase, "collection", None), "output", None)
        if output:
            manifest_path = (root / output.manifest_path).resolve()
            sequence_manifests.append(manifest_path)
            files.append(manifest_path)
            directories.append(
                manifest_path.with_name(manifest_path.name + ".parts").resolve()
            )
    directories = unique_paths(
        directories, "training output directories must be distinct"
    )
    outputs = unique_paths(
        [*files, *phase_files], "training file destinations must be distinct"
    )
    require_file_destinations(outputs)
    validate_path_conflicts(outputs, (), label="training output")
    require_directory_destinations(directories)
    files = [Path(path).resolve() for path in files]
    # A generated snapshot may be selected as the rolling checkpoint on resume.
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
    )
    if group:
        validate_path_conflicts(
            [*sequence_manifests, *directories], protected, label="phase group output"
        )
    elif snapshot_resume:
        external_files = [path for path in files if path != checkpoint]
        validate_path_conflicts(
            [*external_files, *directories], protected, label="training output"
        )
        validate_path_conflicts(
            [checkpoint],
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


def resolve_training_output_paths(run):
    root = Path(run.project_root)
    output_dir = Path("outputs/training") / run.model_name
    model = (root / (run.outputs.model_path or f"models/{run.model_name}.pt")).resolve()
    if model.suffix.lower() != ".pt":
        raise ValueError("run.outputs.model_path must name a .pt portable model")
    run.outputs.model_path = str(model)
    run.outputs.report_path = str(
        (root / (run.outputs.report_path or output_dir / "report.json")).resolve()
    )
    run.outputs.phase_directory = str(
        (root / (run.outputs.phase_directory or output_dir / "phases")).resolve()
    )
