"""Content identities for inputs consumed by training."""

import json
from pathlib import Path

from sequifier.trajectory.sources import file_checksum


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

    def include(self, checksums):
        for path, checksum in checksums.items():
            path = Path(path).absolute()
            resolved = path.resolve()
            self._paths[path] = resolved
            previous = self.identity.setdefault(str(resolved), checksum)
            if previous != checksum:
                raise ValueError(
                    f"Training dependency changed during preparation: {path}"
                )

    def add(self, path):
        resolved = Path(path).resolve()
        if str(resolved) not in self.identity:
            self.include({path: file_checksum(resolved)})
        else:
            self._paths[Path(path).absolute()] = resolved

    def add_reference(self, path, checkpoint_checksum):
        resolved = str(path.resolve())
        if checkpoint_checksum is None and resolved in self._embedded_references:
            # A later phase can still need this same file to create its reference.
            self.include({path: file_checksum(path)})
            self._embedded_references.remove(resolved)
            return
        try:
            self.add(path)
        except FileNotFoundError:
            if checkpoint_checksum is None:
                raise
            self.include({path: checkpoint_checksum})
            self._embedded_references.add(resolved)

    def add_media_directory(self, directory, files):
        self.roots.add(directory.resolve())
        self._media[directory] = tuple(file.name for file in files)
        for path in files:
            self.add(path)

    def merge(self, other):
        self.include(other.identity)
        self.roots.update(other.roots)
        self._media.update(other._media)
        self._paths.update(other._paths)
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
    if view.preconditioning_image:
        media.extend(view.preconditioning_image.views.values())
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
            manifest.add(root / part.metadata_config_path)
            for split in (part.training_data_path, part.validation_data_path):
                if split is None:
                    continue
                path = root / split
                if part.storage_form == "file":
                    manifest.add(path)
                else:
                    manifest.roots.add(path.resolve())
                    metadata = path / "metadata.json"
                    manifest.add(metadata)
                    for shard in json.loads(metadata.read_text())["batch_files"]:
                        manifest.add(path / shard["path"])
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
    for path in config.metadata_paths:
        manifest.add(path)
    for index, phase in enumerate(config.phases):
        manifest.include(prepared.phase_checksums(config, phase, inputs))
        reference = getattr(phase, "reference", None)
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

    repository = config.model.backbone.repository
    if repository is not None and revision_id is not None:
        directory = (
            resolve_repository_path(repository.path, config.project_root)
            / "revisions"
            / revision_id
        )
        manifest.add(directory / "manifest.json")
        manifest.add(directory / "weights.pt")
