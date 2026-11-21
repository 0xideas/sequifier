"""Content identities for inputs consumed by training."""

import hashlib
import json
from pathlib import Path

from sequifier.fingerprints import file_checksum


class DependencyManifest:
    def __init__(self):
        self.identity: dict[str, str] = {}
        self.roots: set[Path] = set()
        self._media: dict[Path, tuple[str, ...]] = {}
        self._paths: dict[Path, Path] = {}
        self._embedded_references: set[str] = set()

    @property
    def protected_paths(self):
        return self.roots | {Path(path) for path in self.identity}

    def _record(self, path, resolved, checksum):
        if self._paths.setdefault(path, resolved) != resolved:
            raise ValueError(f"Training dependency moved: {path}")
        if self.identity.setdefault(str(resolved), checksum) != checksum:
            raise ValueError(f"Training dependency changed during preparation: {path}")

    def add(self, path, checksum=None):
        path = Path(path).absolute()
        resolved = path.resolve()
        if checksum is None:
            checksum = self.identity.get(str(resolved)) or file_checksum(resolved)
        self._record(path, resolved, checksum)
        return checksum

    def read_json(self, path: Path):
        """Track the exact bytes used to interpret a manifest."""
        content = path.read_bytes()
        self.add(path, hashlib.sha256(content).hexdigest())
        return json.loads(content)

    def include(self, checksums):
        for path, checksum in checksums.items():
            self.add(path, checksum)

    def include_files(self, other, paths):
        for path in paths:
            path = Path(path).absolute()
            resolved = other._paths[path]
            self._record(path, resolved, other.identity[str(resolved)])

    def add_reference(self, path, checkpoint_checksum):
        resolved = str(path.resolve())
        if checkpoint_checksum is None and resolved in self._embedded_references:
            # A later phase can still need this same file to create its reference.
            self.add(path, file_checksum(path))
            self._embedded_references.remove(resolved)
            return
        try:
            self.add(path)
        except FileNotFoundError:
            if checkpoint_checksum is None:
                raise
            self.add(path, checkpoint_checksum)
            self._embedded_references.add(resolved)

    def add_prepared_split(
        self, path: Path, *, storage_form=None, include_media=False
    ) -> tuple[Path, ...]:
        """Resolve the split's physical inputs and record what selected them."""
        from sequifier.config.metadata import validate_persisted_metadata_versions
        from sequifier.io.indexed_sequences import IndexedView, prepared_split_form

        form = storage_form or prepared_split_form(path)
        if form == "folder":
            self.roots.add(path.resolve())
            metadata = self.read_json(path / "metadata.json")
            validate_persisted_metadata_versions(metadata)
            files = tuple(path / shard["path"] for shard in metadata["batch_files"])
        elif form == "indexed":
            self.include(IndexedView.load(path).verify(include_media=include_media))
            return (path,)
        else:
            files = (path,)
        for file in files:
            self.add(file)
        return files

    def add_media_directory(self, directory, files):
        self.roots.add(directory.resolve())
        self._media[directory] = tuple(file.name for file in files)
        for path in files:
            self.add(path)

    def merge(self, other):
        self.include_files(other, other._paths)
        self.roots.update(other.roots)
        self._media.update(other._media)
        self._embedded_references.update(other._embedded_references)

    def validate(self, expected):
        if self.identity != expected:
            changed = sorted(
                path
                for path in self.identity.keys() | expected.keys()
                if self.identity.get(path) != expected.get(path)
            )
            raise ValueError(
                f"Training dependencies differ from checkpoint: {changed[:10]}"
            )

    def revalidate(self):
        from sequifier.io.images import media_files

        for directory, files in self._media.items():
            current = media_files(str(directory), str(directory.parent))
            if tuple(file.name for file in current) != files:
                raise ValueError(f"Training media directory changed: {directory}")
        for path, resolved in self._paths.items():
            if path.resolve() != resolved:
                raise ValueError(f"Training dependency moved: {path}")
        for path, checksum in self.identity.items():
            try:
                current = file_checksum(path)
            except FileNotFoundError:
                if path in self._embedded_references:
                    continue
                raise
            if current != checksum:
                raise ValueError(f"Training dependency changed: {path}")


def add_initialization(manifest, config, incoming=None):
    if path := incoming or config.initialize_from:
        manifest.add(Path(config.project_root) / path)


def _add_loader_media(manifest, view):
    from sequifier.io.images import media_files

    root = Path(view.project_root)
    if view.static_columns and view.static_features_path:
        manifest.add(root / view.static_features_path)
    media = []
    for source in view.image_sources.values():
        if source.path:
            media.append(source.path)
        if source.time_index_path:
            manifest.add(root / source.time_index_path)
        for path in source.video_index_paths.values():
            manifest.add(root / path)
    if view.preconditioning_image and view.preconditioning_image.paths:
        media.extend(view.preconditioning_image.paths.values())
    for path in media:
        manifest.add_media_directory(root / path, media_files(path, view.project_root))


def supervised_dependencies(config, incoming=None):
    from sequifier.config.train_config import dataset_part_view

    manifest = DependencyManifest()
    root = Path(config.project_root)
    add_initialization(manifest, config, incoming)
    for path in config.model_metadata_paths.values():
        manifest.add(path)
    for name, dataset in config.dataset_training.items():
        for part_name, part in dataset.parts.items():
            if part.metadata_from_file:
                manifest.add(root / part.metadata_config_path)
            for split in set(part.split_paths.values()):
                manifest.add_prepared_split(Path(split), storage_form=part.storage_form)
            _add_loader_media(
                manifest,
                dataset_part_view(
                    config, name, part_name, batch_size=1, window_stride=None
                ),
            )
    return manifest


def policy_dependencies(
    config, prepared, inputs, *, saved=None, committed_reference_count=0
):
    manifest = DependencyManifest()
    add_initialization(manifest, config)
    if path := config.parent.model_metadata_paths.get(config.policy.interface_name):
        manifest.add(path)
    for index, phase in enumerate(config.phases):
        _, _, data = prepared.phase_dependencies(
            config.parent.policy_data, phase, inputs
        )
        manifest.merge(data)
        reference = phase.reference_config
        if reference and reference.path:
            path = Path(config.project_root) / reference.path
            checksum = (
                saved.get(str(path.resolve()))
                if saved is not None and index < committed_reference_count
                else None
            )
            manifest.add_reference(path, checksum)
    return manifest


def add_backbone_revision(manifest, config, revision_id):
    from sequifier.artifacts.backbone_repository import resolve_repository_path

    repository = config.execution.backbone.repository
    if repository is not None and revision_id is not None:
        directory = (
            resolve_repository_path(repository.path, config.project_root)
            / "revisions"
            / revision_id
        )
        manifest.add(directory / "manifest.json")
        manifest.add(directory / "weights.pt")
