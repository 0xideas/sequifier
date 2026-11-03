"""Validate original image media and read aligned frames without rewriting it."""

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from sequifier.config.image_config import ImageSourceConfig, PreconditioningImageConfig
from sequifier.helpers import normalize_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _files(path: str, project_root: str) -> list[Path]:
    directory = Path(normalize_path(path, project_root))
    if not directory.is_dir():
        raise ValueError(f"Image source is not a directory: {directory}")
    return sorted(
        file
        for file in directory.iterdir()
        if file.is_file() and not file.name.startswith(".")
    )


def _signature(path: str, project_root: str) -> str:
    digest = hashlib.sha256()
    for file in _files(path, project_root):
        stat = file.stat()
        digest.update(f"{file.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _preconditioning_signature(
    source: PreconditioningImageConfig, project_root: str
) -> str:
    digest = hashlib.sha256()
    for name, path in sorted(source.views.items()):
        digest.update(f"{name}\0{_signature(path, project_root)}\n".encode())
    return digest.hexdigest()


def _index(
    path: str, project_root: str, *, kind: str, preconditioning: bool = False
) -> dict[int, dict[int, Path]]:
    extensions = IMAGE_EXTENSIONS if kind == "images" else VIDEO_EXTENSIONS
    result: dict[int, dict[int, Path]] = {}
    for file in _files(path, project_root):
        if file.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported file in image source: {file}")
        stem = file.stem
        try:
            if preconditioning or kind == "videos":
                sequence_id, frame_index = int(stem), 0
            else:
                sequence, separator, frame = stem.rpartition("-")
                if not separator:
                    raise ValueError
                sequence_id, frame_index = int(sequence), int(frame)
        except ValueError as error:
            raise ValueError(f"Invalid image source filename: {file}") from error
        if frame_index < 0:
            raise ValueError(f"Negative frame index in {file}")
        frames = result.setdefault(sequence_id, {})
        if frame_index in frames:
            raise ValueError(
                f"Duplicate image coordinate in {file} and {frames[frame_index]}"
            )
        frames[frame_index] = file
    return result


def _video_frame_count(path: Path) -> int:
    try:
        import av
    except ImportError as error:
        raise ImportError(
            "Video sources require the sequifier[images] extra"
        ) from error
    try:
        with av.open(str(path)) as container:
            return sum(1 for _ in container.decode(video=0))
    except Exception as error:
        raise ValueError(f"Cannot decode video {path}: {error}") from error


def validate_sources(
    sources: dict[str, ImageSourceConfig],
    preconditioning: PreconditioningImageConfig | None,
    sequence_lengths: dict[int, int],
    project_root: str,
) -> dict[str, Any]:
    """Return a small source manifest after checking every original image/frame."""
    from PIL import Image

    if not sequence_lengths:
        raise ValueError("Image sources require at least one sequence")
    manifest: dict[str, Any] = {"image_sources": {}, "preconditioning_image": None}
    for name, source in sources.items():
        index = _index(source.path, project_root, kind=source.type)
        if set(index) != set(sequence_lengths):
            raise ValueError(
                f"Image view {name!r} sequence IDs differ from tabular data: "
                f"missing={sorted(set(sequence_lengths) - set(index))[:10]}, "
                f"extra={sorted(set(index) - set(sequence_lengths))[:10]}"
            )
        for sequence_id, length in sequence_lengths.items():
            files = index[sequence_id]
            expected = source.alignment.expected_frames(length)
            if source.type == "images":
                if set(files) != set(range(expected)):
                    raise ValueError(
                        f"Image view {name!r}, sequence {sequence_id}: expected "
                        f"frame indices 0..{expected - 1}, found {sorted(files)}"
                    )
                for file in files.values():
                    try:
                        with Image.open(file) as image:
                            image.verify()
                    except Exception as error:
                        raise ValueError(
                            f"Cannot decode image {file}: {error}"
                        ) from error
            else:
                if set(files) != {0}:
                    raise ValueError(
                        f"Video view {name!r}, sequence {sequence_id} needs one video"
                    )
                actual = _video_frame_count(files[0])
                if actual != expected:
                    raise ValueError(
                        f"Video view {name!r}, sequence {sequence_id}: "
                        f"expected {expected} decoded frames, found {actual}"
                    )
        manifest["image_sources"][name] = {
            **source.model_dump(mode="json"),
            "signature": _signature(source.path, project_root),
        }
    if preconditioning is not None:
        for name, path in preconditioning.views.items():
            index = _index(path, project_root, kind="images", preconditioning=True)
            if set(index) != set(sequence_lengths):
                raise ValueError(
                    f"Preconditioning view {name!r} requires one image per sequence"
                )
            for frames in index.values():
                try:
                    with Image.open(frames[0]) as image:
                        image.verify()
                except Exception as error:
                    raise ValueError(
                        f"Cannot decode preconditioning image {frames[0]}: {error}"
                    ) from error
        manifest["preconditioning_image"] = {
            **preconditioning.model_dump(mode="json"),
            "signature": _preconditioning_signature(preconditioning, project_root),
        }
    return manifest


class ImageStore:
    """Worker-local media lookup with a bounded decoded-frame cache."""

    def __init__(
        self,
        sources: dict[str, ImageSourceConfig],
        preconditioning: PreconditioningImageConfig | None,
        project_root: str,
    ) -> None:
        self.sources = sources
        self.preconditioning = preconditioning
        for name, source in sources.items():
            if (
                source.signature is not None
                and _signature(source.path, project_root) != source.signature
            ):
                raise ValueError(f"Image view {name!r} changed after preprocessing")
        if (
            preconditioning is not None
            and preconditioning.signature is not None
            and _preconditioning_signature(preconditioning, project_root)
            != preconditioning.signature
        ):
            raise ValueError("Preconditioning images changed after preprocessing")
        self.indices = {
            name: _index(source.path, project_root, kind=source.type)
            for name, source in sources.items()
        }
        self.prefix_indices = (
            {
                name: _index(path, project_root, kind="images", preconditioning=True)
                for name, path in sorted(preconditioning.views.items())
            }
            if preconditioning is not None
            else None
        )
        self.cache: OrderedDict[tuple[str, int, int, int], torch.Tensor] = OrderedDict()

    def _transform(self, image: Any, source: Any) -> torch.Tensor:
        from PIL import Image

        image = image.convert("RGB")
        width, height = image.size
        scale = source.resize_size / min(width, height)
        image = image.resize(
            (round(width * scale), round(height * scale)), Image.Resampling.BILINEAR
        )
        crop = source.crop_size
        left = (image.width - crop) // 2
        top = (image.height - crop) // 2
        image = image.crop((left, top, left + crop, top + crop))
        import numpy as np

        tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
        return tensor

    def _cache_frame(self, path: Path, frame: int, source: Any, image: Any) -> None:
        key = (str(path), frame, source.resize_size, source.crop_size)
        self.cache[key] = self._transform(image, source)
        if len(self.cache) > 128:
            self.cache.popitem(last=False)

    def _prime_video(
        self, path: Path, frames: set[int], source: Any
    ) -> dict[int, torch.Tensor]:
        import av

        found = {
            frame: self.cache[(str(path), frame, source.resize_size, source.crop_size)]
            for frame in frames
            if (str(path), frame, source.resize_size, source.crop_size) in self.cache
        }
        missing = {frame for frame in frames if frame not in found}
        if not missing:
            return found
        with av.open(str(path)) as container:
            for index, decoded in enumerate(container.decode(video=0)):
                if index in missing:
                    self._cache_frame(path, index, source, decoded.to_image())
                    key = (str(path), index, source.resize_size, source.crop_size)
                    found[index] = self.cache[key]
                    missing.remove(index)
                if not missing:
                    break
        if missing:
            raise ValueError(f"Missing decoded frames {sorted(missing)[:10]} in {path}")
        return found

    def _decode(self, path: Path, frame: int, source: Any) -> torch.Tensor:
        from PIL import Image, ImageOps

        key = (str(path), frame, source.resize_size, source.crop_size)
        if key not in self.cache:
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                return self._prime_video(path, {frame}, source)[frame]
            else:
                with Image.open(path) as opened:
                    self._cache_frame(
                        path, frame, source, ImageOps.exif_transpose(opened)
                    )
        self.cache.move_to_end(key)
        return self.cache[key]

    def gather(
        self,
        sequence_ids: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        inputs: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        batch, time = positions.shape
        for name, source in self.sources.items():
            capacity = source.alignment.max_frames_per_item
            images = torch.zeros(
                batch,
                time,
                capacity,
                3,
                source.crop_size,
                source.crop_size,
                dtype=torch.uint8,
            )
            mask = torch.zeros(batch, time, capacity, dtype=torch.bool)
            requests = []
            videos: dict[Path, set[int]] = {}
            for row in range(batch):
                sequence_id = int(sequence_ids[row])
                for step in range(time):
                    if not bool(valid[row, step]):
                        continue
                    item = int(positions[row, step])
                    if item < 0:
                        raise ValueError("Image item positions must start at zero")
                    for slot, frame in enumerate(source.alignment.frame_indices(item)):
                        path = self.indices[name][sequence_id][
                            frame if source.type == "images" else 0
                        ]
                        media_frame = frame if source.type == "videos" else 0
                        requests.append((row, step, slot, path, media_frame))
                        if source.type == "videos":
                            videos.setdefault(path, set()).add(media_frame)
            video_frames = {
                path: self._prime_video(path, frames, source)
                for path, frames in videos.items()
            }
            for row, step, slot, path, frame in requests:
                images[row, step, slot] = (
                    video_frames[path][frame]
                    if source.type == "videos"
                    else self._decode(path, frame, source)
                )
                mask[row, step, slot] = True
            inputs[f"__image:{name}"] = images
            masks[f"__image_mask:{name}"] = mask
        if self.prefix_indices is not None and self.preconditioning is not None:
            crop = self.preconditioning.crop_size
            prefix = torch.zeros(
                batch, len(self.prefix_indices), 3, crop, crop, dtype=torch.uint8
            )
            for row in range(batch):
                for view, index in enumerate(self.prefix_indices.values()):
                    prefix[row, view] = self._decode(
                        index[int(sequence_ids[row])][0], 0, self.preconditioning
                    )
            inputs["__preconditioning_image"] = prefix
        return inputs, masks
