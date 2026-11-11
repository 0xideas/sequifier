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
