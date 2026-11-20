"""Resolve sequence source files, manifests, and declared formats."""

from dataclasses import dataclass
from pathlib import Path

from sequifier.artifacts.dependencies import DependencyManifest
from sequifier.config.policy_train_config import (
    SequenceSourceConfig,
    resolve_sequence_source,
)
from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.trajectory.store import load_trajectory


def _source_files(source, dependencies):
    manifest = None
    if source.manifest_path is None:
        paths = [Path(path) for path in source.paths]
    else:
        manifest_path = Path(source.manifest_path)
        manifest = dependencies.read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError("Source manifest must be a mapping")
        manifest_format = manifest.get("format")
        if manifest_format == "sequifier.sequence-data":
            require_contract_version(
                manifest_format,
                manifest.get("version"),
                ContractVersions.SEQUENCE_DATA_MANIFEST,
            )
            entries_field = "shards"
        elif manifest_format == "sequifier.trajectory-source":
            if "version" in manifest:
                raise ValueError(
                    "Authored trajectory-source manifests must omit version"
                )
            entries_field = "episodes"
        else:
            raise ValueError(f"Unsupported source manifest format {manifest_format!r}")
        entries = manifest.get(entries_field)
        if not isinstance(entries, list) or not entries:
            raise ValueError(
                f"source manifest requires a nonempty {entries_field} list"
            )
        paths = []
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("path"), str)
                or not entry["path"]
            ):
                raise ValueError("each manifest entry requires a nonempty path")
            path = manifest_path.parent / entry["path"]
            if manifest_format == "sequifier.sequence-data" and dependencies.add(
                path
            ) != entry.get("sha256"):
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
    return [path.resolve() for path in _source_files(source, DependencyManifest())[0]]


@dataclass(frozen=True)
class ResolvedSequenceSource:
    source: SequenceSourceConfig
    paths: tuple[Path, ...]
    kind: str
    dependencies: DependencyManifest

    def provenance(self, declaration):
        """Confirm the published dataset still describes the inputs we read."""
        self.dependencies.revalidate()
        return {
            "source": declaration.model_dump(mode="json"),
            "files": [
                {"path": path, "sha256": checksum}
                for path, checksum in sorted(self.dependencies.identity.items())
            ],
        }


def resolve_source(source: SequenceSourceConfig) -> ResolvedSequenceSource:
    dependencies = DependencyManifest()
    if source.split_index is not None:
        source = resolve_sequence_source(source, Path.cwd())
    # Retain manifest-relative locations so revalidation also detects retargeted links.
    paths, manifest = _source_files(source, dependencies)
    for path in (source.metadata_config_path, source.static_data_path):
        if path is not None:
            dependencies.add(path)
    if source.format == "prepared":
        paths = [
            file
            for path in paths
            for file in dependencies.add_prepared_split(path, include_media=True)
        ]
        if not paths or len({path.resolve() for path in paths}) != len(paths):
            raise ValueError("prepared splits require nonempty, unique shard paths")
    else:
        for path in paths:
            dependencies.add(path)
    resolved_paths = tuple(path.resolve() for path in paths)
    if manifest is not None and manifest.get("format") == "sequifier.sequence-data":
        if source.encoding != "raw" or source.episode_end != "unknown":
            raise ValueError(
                "committed sequence data already declares its encoding and boundaries"
            )
        source = source.model_copy(update={"format": "table"})
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
    return ResolvedSequenceSource(source, resolved_paths, kind, dependencies)
