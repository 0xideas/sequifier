"""Resolve sequence source files, manifests, and declared formats."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from sequifier.config.policy_train_config import (
    PolicyTableColumns,
    SequenceSourceConfig,
    resolve_sequence_source,
)
from sequifier.trajectory.store import load_trajectory


def file_checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checksum(path, checksums):
    path = Path(path)
    if path not in checksums:
        checksums[path] = file_checksum(path)
    return checksums[path]


def _source_files(source, checksums):
    manifest = None
    if source.manifest_path is None:
        paths = [Path(path) for path in source.paths]
    else:
        manifest_path = Path(source.manifest_path)
        content = manifest_path.read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        if manifest_path in checksums and checksums[manifest_path] != checksum:
            raise ValueError(
                f"source manifest changed during resolution: {manifest_path}"
            )
        checksums[manifest_path] = checksum
        manifest = json.loads(content.decode("utf-8"))
        if not isinstance(manifest, dict) or manifest.get("version") != 1:
            raise ValueError("source manifest must have version 1")
        sequence_manifest = manifest.get("format") == "sequifier.sequence-data"
        entries = manifest.get("shards" if sequence_manifest else "episodes")
        if not isinstance(entries, list) or not entries:
            raise ValueError("source manifest requires a nonempty episodes list")
        paths = []
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("path"), str)
                or not entry["path"]
            ):
                raise ValueError("each manifest entry requires a nonempty path")
            path = manifest_path.parent / entry["path"]
            if sequence_manifest and _checksum(path, checksums) != entry.get("sha256"):
                raise ValueError(f"sequence shard checksum disagrees with {path}")
            if (
                "episode_id" in entry
                and load_trajectory(path).episode_id != entry["episode_id"]
            ):
                raise ValueError(f"manifest episode ID disagrees with {path}")
            paths.append(path)
    if not paths or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("sources require nonempty, unique paths")
    return paths, manifest


def source_paths(source: SequenceSourceConfig) -> list[Path]:
    return [path.resolve() for path in _source_files(source, {})[0]]


@dataclass(frozen=True)
class ResolvedSequenceSource:
    source: SequenceSourceConfig
    paths: tuple[Path, ...]
    kind: str
    checksums: dict[Path, str]


def resolve_source(
    source: SequenceSourceConfig, *, checksums: dict[Path, str] | None = None
) -> ResolvedSequenceSource:
    checksums = {} if checksums is None else checksums
    if source.split_index is not None:
        source = resolve_sequence_source(source, Path.cwd())
    # Retain manifest-relative locations so revalidation also detects retargeted links.
    paths, manifest = _source_files(source, checksums)
    dependencies = set(paths)
    dependencies.update(
        Path(path)
        for path in (
            source.manifest_path,
            source.metadata_config_path,
            source.static_data_path,
        )
        if path
    )
    verified = {path: _checksum(path, checksums) for path in dependencies}
    resolved_paths = tuple(path.resolve() for path in paths)
    if manifest is not None and manifest.get("format") == "sequifier.sequence-data":
        recorded_columns = PolicyTableColumns.model_validate(manifest["columns"])
        if (
            source.columns != PolicyTableColumns()
            and source.columns != recorded_columns
        ):
            raise ValueError(
                "source mappings disagree with the committed sequence manifest"
            )
        if source.encoding != "raw" or source.episode_end != "unknown":
            raise ValueError(
                "committed sequence data already declares its encoding and boundaries"
            )
        source = source.model_copy(
            update={"columns": recorded_columns, "format": "table"}
        )
    kind = source.format
    if kind == "auto":
        suffixes = {path.suffix.lower() for path in resolved_paths}
        if suffixes <= {".json"}:
            kind = "trajectory"
        elif suffixes <= {".csv", ".parquet"}:
            kind = "table"
        elif suffixes == {".pt"}:
            kind = "prepared"
        else:
            raise ValueError(
                "select a source format explicitly for mixed or unsupported files"
            )
    return ResolvedSequenceSource(source, resolved_paths, kind, verified)
