"""Resolve sequence source files, manifests, and declared formats."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from sequifier.artifacts.dependencies import DependencyManifest
from sequifier.config.metadata import DatasetMetadata, load_dataset_metadata
from sequifier.config.policy_train_config import SequenceSourceConfig
from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.fingerprints import fingerprint

if TYPE_CHECKING:
    from sequifier.io.images import ImageStore


def _source_files(source, root, metadata, dependencies):
    manifest = None
    if source.split is not None:
        index = metadata.named_split_index(source.split)
        paths = [root / metadata.split_paths[index]]
    elif source.manifest_path is None:
        paths = [root / path for path in source.paths]
    else:
        manifest_path = root / source.manifest_path
        manifest = dependencies.read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError("Source manifest must be a mapping")
        manifest_format = manifest.get("format")
        if manifest_format != "sequifier.sequence-data":
            raise ValueError(f"Unsupported source manifest format {manifest_format!r}")

        require_contract_version(
            manifest_format,
            manifest.get("version"),
            ContractVersions.SEQUENCE_DATA_MANIFEST,
        )
        entries = manifest.get("shards")
        if not isinstance(entries, list) or not entries:
            raise ValueError("source manifest requires a nonempty shards list")
        paths = []
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("path"), str)
                or not entry["path"]
            ):
                raise ValueError("each manifest entry requires a nonempty path")
            path = manifest_path.parent / entry["path"]
            if dependencies.add(path) != entry.get("sha256"):
                raise ValueError(f"sequence shard checksum disagrees with {path}")
            paths.append(path)
    if not paths or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("sources require nonempty, unique paths")
    return paths, manifest


@dataclass(frozen=True)
class ResolvedSequenceSource:
    declaration: SequenceSourceConfig
    project_root: Path
    paths: tuple[Path, ...]
    kind: str
    metadata_path: Path | None
    metadata: DatasetMetadata | None
    static_path: Path | None
    dependencies: DependencyManifest
    manifest: dict | None = None
    image_store: "ImageStore | None" = None

    @cached_property
    def cache_key(self) -> str:
        return fingerprint(
            {
                "source": self.declaration.model_dump(mode="json"),
                "project_root": str(self.project_root),
                "paths": [str(path) for path in self.paths],
                "kind": self.kind,
                "checksums": self.dependencies.identity,
            }
        )

    def provenance(self):
        """Confirm the published dataset still describes the inputs we read."""
        self.dependencies.revalidate()
        return {
            "source": self.declaration.model_dump(mode="json"),
            "project_root": str(self.project_root),
            "files": [
                {"path": path, "sha256": checksum}
                for path, checksum in sorted(self.dependencies.identity.items())
            ],
        }


def resolve_source(
    source: SequenceSourceConfig,
    project_root: str | Path = ".",
    *,
    metadata: DatasetMetadata | None = None,
    prepared: bool = False,
) -> ResolvedSequenceSource:
    root = Path(project_root).resolve()
    dependencies = DependencyManifest()
    metadata_path = (
        root / source.metadata_config_path if source.metadata_config_path else None
    )
    static_path = root / source.static_data_path if source.static_data_path else None
    if metadata_path is not None:
        dependencies.add(metadata_path)
        if metadata is None:
            metadata = load_dataset_metadata(str(metadata_path))
    # Keep manifest-relative locations for symlink revalidation.
    paths, manifest = _source_files(source, root, metadata, dependencies)
    kind = "prepared" if prepared or source.split is not None else source.format
    if manifest is not None:
        assert source.manifest_path is not None
        if source.encoding != "raw" or source.episode_end != "unknown":
            raise ValueError(
                "committed sequence data already declares its encoding and boundaries"
            )
        entry = manifest["sequence_metadata"]
        static_path = (root / source.manifest_path).parent / entry["path"]
        if dependencies.add(static_path) != entry["sha256"]:
            raise ValueError("Sequence metadata checksum mismatch")
        kind = "table"
    if kind == "auto":
        suffixes = {path.resolve().suffix.lower() for path in paths}
        if suffixes <= {".json"}:
            from sequifier.io.indexed_sequences import is_indexed_view

            if any(is_indexed_view(path) for path in paths):
                if len(paths) != 1:
                    raise ValueError(
                        "An indexed sequence source requires exactly one split manifest"
                    )
                kind = "prepared"
            else:
                raise ValueError("Use manifest_path for a trajectory dataset")
        elif suffixes <= {".csv", ".parquet"}:
            kind = "table"
        elif suffixes == {".pt"}:
            kind = "prepared"
        else:
            raise ValueError(
                "select a source format explicitly for mixed or unsupported files"
            )
    if kind == "prepared":
        if metadata and metadata.source_observations_path:
            dependencies.add(root / metadata.source_observations_path)
        if (
            static_path is None
            and metadata is not None
            and metadata.static_features_path
        ):
            static_path = root / metadata.static_features_path
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
    if static_path is not None:
        dependencies.add(static_path)
    return ResolvedSequenceSource(
        source,
        root,
        tuple(path.resolve() for path in paths),
        kind,
        metadata_path,
        metadata,
        static_path,
        dependencies,
        manifest,
    )
