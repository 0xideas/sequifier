"""Validate original image media and read aligned frames without rewriting it."""

import hashlib
import json
import os
import re
import tempfile
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, BinaryIO

import polars as pl
import torch

from sequifier.config.image_config import ImageSourceConfig, PreconditioningImageConfig
from sequifier.helpers import normalize_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
_IMAGE_NAME = re.compile(r"^(-?\d+)-(\d+)(?:-(\d{9}|\d{17})(?:-(\d+))?)?$")
_VIDEO_NAME = re.compile(r"^(-?\d+)(?:-(\d{9}|\d{17}))?$")


@dataclass(frozen=True)
class _VideoIndex:
    pts: tuple[int, ...]
    keyframes: tuple[int, ...]
    seek_pts: tuple[int, ...]
    by_pts: dict[int, int]


_VIDEO_INDEX_VERSION = 1


def _media_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _civil_milliseconds(value: datetime) -> int:
    delta = value.replace(tzinfo=None) - datetime(1970, 1, 1)
    return delta.days * 86_400_000 + delta.seconds * 1000 + delta.microseconds // 1000


def _time_ms(value: Any, *, absolute: bool) -> int:
    """Convert a civil timestamp or a clock duration to milliseconds.

    Absolute timestamps are compared as written; no timezone conversion occurs.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value) if absolute else f"{value:09d}"
    if isinstance(value, datetime):
        if not absolute or value.microsecond % 1000:
            raise ValueError(f"Invalid millisecond timestamp: {value!r}")
        return _civil_milliseconds(value)
    if isinstance(value, date):
        if not absolute:
            raise ValueError(f"Invalid relative time: {value!r}")
        return _civil_milliseconds(datetime.combine(value, datetime.min.time()))
    if not isinstance(value, str):
        raise ValueError(f"Invalid time value: {value!r}")
    if absolute:
        if len(value) == 17 and value.isdigit():
            parsed = datetime.strptime(value, "%Y%m%d%H%M%S%f")
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            parsed = datetime.combine(date.fromisoformat(value), datetime.min.time())
        else:
            if not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\.\d{3,6}(?:Z|[+-]\d{2}:\d{2})?",
                value,
            ):
                raise ValueError(f"Absolute timestamp needs milliseconds: {value!r}")
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(f"Invalid absolute timestamp: {value!r}") from error
        if parsed.microsecond % 1000:
            raise ValueError(f"Timestamp needs millisecond precision: {value!r}")
        return _civil_milliseconds(parsed)
    if len(value) != 9 or not value.isdigit():
        raise ValueError(f"Relative time must be HHmmssSSS: {value!r}")
    hours, minutes, seconds, millis = (
        int(value[:2]),
        int(value[2:4]),
        int(value[4:6]),
        int(value[6:]),
    )
    if hours > 23 or minutes > 59 or seconds > 59:
        raise ValueError(f"Invalid relative time: {value!r}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


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


def _item_times_signature(item_times: dict[int, list[int]]) -> str:
    digest = hashlib.sha256()
    for sequence_id, times in sorted(item_times.items()):
        digest.update(f"{sequence_id}:".encode())
        for stamp in times:
            digest.update(f"{stamp},".encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _index(
    path: str, project_root: str, *, kind: str, preconditioning: bool = False
) -> tuple[
    dict[int, dict[int, Path]],
    dict[int, dict[int, tuple[str | None, int | None]]],
    str,
]:
    extensions = IMAGE_EXTENSIONS if kind == "images" else VIDEO_EXTENSIONS
    result: dict[int, dict[int, Path]] = {}
    timing: dict[int, dict[int, tuple[str | None, int | None]]] = {}
    for file in _files(path, project_root):
        if file.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported file in image source: {file}")
        if preconditioning:
            match = re.fullmatch(r"-?\d+", file.stem)
            if match is None:
                raise ValueError(f"Invalid preconditioning image filename: {file}")
            sequence_id, frame_index, stamp, interval = int(file.stem), 0, None, None
        elif kind == "videos":
            match = _VIDEO_NAME.fullmatch(file.stem)
            if match is None:
                raise ValueError(f"Invalid video filename: {file}")
            sequence_id, frame_index = int(match[1]), 0
            stamp, interval = match[2], None
        else:
            match = _IMAGE_NAME.fullmatch(file.stem)
            if match is None:
                raise ValueError(f"Invalid image filename: {file}")
            sequence_id, frame_index = int(match[1]), int(match[2])
            stamp, interval = match[3], int(match[4]) if match[4] else None
            if interval is not None and (frame_index != 0 or interval <= 0):
                raise ValueError(f"Frame interval belongs only on frame 0: {file}")
        frames = result.setdefault(sequence_id, {})
        if frame_index in frames:
            raise ValueError(
                f"Duplicate image coordinate in {file} and {frames[frame_index]}"
            )
        frames[frame_index] = file
        timing.setdefault(sequence_id, {})[frame_index] = (stamp, interval)
    modes = set()
    representations = set()
    for sequence_id, frames in timing.items():
        if preconditioning:
            modes.add("none")
            continue
        entries = [frames[i] for i in sorted(frames)]
        if kind == "videos":
            mode = "all" if entries[0][0] is not None else "none"
        elif entries[0][1] is not None and all(
            stamp is None and interval is None for stamp, interval in entries[1:]
        ):
            mode = "first"
        elif all(stamp is not None and interval is None for stamp, interval in entries):
            mode = "all"
        elif all(stamp is None and interval is None for stamp, interval in entries):
            mode = "none"
        else:
            raise ValueError(
                f"Mixed timestamp filename patterns for sequence {sequence_id}"
            )
        modes.add(mode)
        representations.update(len(stamp) for stamp, _ in entries if stamp is not None)
    if len(modes) > 1 or len(representations) > 1:
        raise ValueError(
            "Timestamp filename patterns must be uniform across sequence IDs"
        )
    mode = next(iter(modes), "none")
    representation = next(iter(representations), None)
    time_kind = "relative" if representation == 9 else "absolute"
    return result, timing, "none" if mode == "none" else f"{mode}:{time_kind}"


def _scan_video(path: Path) -> dict[str, Any]:
    try:
        import av
    except ImportError as error:
        raise ImportError(
            "Video sources require the sequifier[images] extra"
        ) from error
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            base = stream.time_base
            stamps = []
            anchors = []
            seek_points = []
            times = []
            origin = None
            has_all_timestamps = True
            for number, frame in enumerate(container.decode(video=0)):
                stamps.append(frame.pts)
                if frame.key_frame:
                    anchors.append(number)
                    seek_points.append(
                        int(frame.pts * frame.time_base / base)
                        if frame.pts is not None
                        and frame.time_base is not None
                        and base is not None
                        else None
                    )
                frame_time = frame.time
                if frame_time is None:
                    has_all_timestamps = False
                else:
                    if origin is None:
                        origin = frame_time
                    if has_all_timestamps:
                        times.append(round((frame_time - origin) * 1000))
            if not has_all_timestamps:
                rate = stream.average_rate
                times = (
                    [round(index * 1000 / float(rate)) for index in range(len(stamps))]
                    if rate is not None and rate > 0
                    else None
                )
            return {
                "version": _VIDEO_INDEX_VERSION,
                "media": _media_signature(path),
                "time_base": [base.numerator, base.denominator] if base else None,
                "frame_count": len(stamps),
                "pts": stamps,
                "keyframes": anchors,
                "seek_pts": seek_points,
                "frame_times_ms": times,
            }
    except Exception as error:
        raise ValueError(f"Cannot decode video {path}: {error}") from error


def _validated_video_times(path: Path, offsets: list[int] | None) -> list[int]:
    if offsets is None:
        raise ValueError(f"Video {path} needs presentation timestamps or a frame rate")
    if any(b <= a for a, b in zip(offsets, offsets[1:])):
        raise ValueError(f"Video {path} frame timestamps must increase")
    gaps = [b - a for a, b in zip(offsets, offsets[1:])]
    if gaps and max(gaps) - min(gaps) > 1:
        raise ValueError(f"Video {path} frames must be evenly spaced in time")
    return offsets


def _source_frame_times(
    source: ImageSourceConfig,
    files: dict[int, Path],
    stamps: dict[int, tuple[str | None, int | None]],
    filename_mode: str,
    video_offsets: list[int] | None = None,
) -> list[int]:
    mode, representation = filename_mode.split(":")
    absolute = representation == "absolute"
    if source.type == "videos":
        start = _time_ms(stamps[0][0], absolute=absolute)
        offsets = _validated_video_times(files[0], video_offsets)
        times = [start + offset for offset in offsets]
    elif mode == "first":
        start = _time_ms(stamps[0][0], absolute=absolute)
        interval = stamps[0][1]
        assert interval is not None
        times = [start + frame * interval for frame in sorted(files)]
    else:
        times = [
            _time_ms(stamps[frame][0], absolute=absolute) for frame in sorted(files)
        ]
        gaps = [b - a for a, b in zip(times, times[1:])]
        if gaps and max(gaps) - min(gaps) > 1:
            raise ValueError("Timestamped image frames must be evenly spaced")
    if not absolute and times and times[0] != 0:
        raise ValueError("Relative media time must start at 000000000")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Image frame times must increase")
    return times


def validate_sources(
    sources: dict[str, ImageSourceConfig],
    preconditioning: PreconditioningImageConfig | None,
    sequence_lengths: dict[int, int],
    project_root: str,
    item_times: dict[int, list[int]] | None = None,
    time_mode: str | None = None,
    time_index_path: str | None = None,
    video_index_dir: str | None = None,
) -> dict[str, Any]:
    """Return a small source manifest after checking every original image/frame."""
    from PIL import Image

    if not sequence_lengths:
        raise ValueError("Image sources require at least one sequence")
    manifest: dict[str, Any] = {"image_sources": {}, "preconditioning_image": None}
    video_index_paths: dict[str, dict[int, str]] = {}
    for name, source in sources.items():
        if source.path is None:
            raise ValueError(f"Image view {name!r} needs a source path")
        index, timing, filename_mode = _index(
            source.path, project_root, kind=source.type
        )
        timed = filename_mode != "none"
        representation = filename_mode.rsplit(":", 1)[-1] if timed else None
        if timed and (
            item_times is None or time_mode is None or time_index_path is None
        ):
            raise ValueError(f"Image view {name!r} requires time_alignment_column")
        if timed and representation != time_mode:
            raise ValueError(
                f"Image view {name!r} uses {representation} time but "
                f"time_alignment_column uses {time_mode} time"
            )
        if timed and source.max_frames_per_item is None:
            raise ValueError(f"Timed image view {name!r} requires max_frames_per_item")
        if set(index) != set(sequence_lengths):
            raise ValueError(
                f"Image view {name!r} sequence IDs differ from tabular data: "
                f"missing={sorted(set(sequence_lengths) - set(index))[:10]}, "
                f"extra={sorted(set(index) - set(sequence_lengths))[:10]}"
            )
        for sequence_id, length in sequence_lengths.items():
            files = index[sequence_id]
            expected = source.alignment.expected_frames(length)
            record: dict[str, Any] | None = None
            if source.type == "images":
                required = len(files) if timed else expected
                if set(files) != set(range(required)):
                    raise ValueError(
                        f"Image view {name!r}, sequence {sequence_id}: expected "
                        f"frame indices 0..{required - 1}, found {sorted(files)}"
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
                if video_index_dir is None:
                    raise ValueError("Video sources need a generated index directory")
                record = _scan_video(files[0])
                if not timed and (actual := len(record["pts"])) != expected:
                    raise ValueError(
                        f"Video view {name!r}, sequence {sequence_id}: "
                        f"expected {expected} decoded frames, found {actual}"
                    )
            if timed:
                times = _source_frame_times(
                    source,
                    files,
                    timing[sequence_id],
                    filename_mode,
                    record["frame_times_ms"] if record is not None else None,
                )
                if not times:
                    raise ValueError(
                        f"Image view {name!r}, sequence {sequence_id} has no frames"
                    )
                assert item_times is not None
                if (
                    sequence_id not in item_times
                    or len(item_times[sequence_id]) != length
                ):
                    raise ValueError(f"Missing item times for sequence {sequence_id}")
                if times[0] > item_times[sequence_id][0]:
                    raise ValueError(
                        f"Image view {name!r}, sequence {sequence_id} starts after its first item"
                    )
            if record is not None:
                assert video_index_dir is not None
                directory = Path(normalize_path(video_index_dir, project_root))
                directory.mkdir(parents=True, exist_ok=True)
                view_id = hashlib.sha256(name.encode()).hexdigest()[:16]
                relative = f"{video_index_dir}/{view_id}-{sequence_id}.json"
                destination = Path(normalize_path(relative, project_root))
                temporary = destination.with_suffix(".json.tmp")
                with temporary.open("w") as file:
                    json.dump(record, file, separators=(",", ":"))
                os.replace(temporary, destination)
                video_index_paths.setdefault(name, {})[sequence_id] = relative
        manifest["image_sources"][name] = {
            **source.model_dump(mode="json", exclude={"path"}),
            "signature": _signature(source.path, project_root),
            "timing_mode": representation,
            "time_index_path": time_index_path if timed else None,
            "time_index_signature": (
                _item_times_signature(item_times)
                if timed and item_times is not None
                else None
            ),
            "video_index_paths": video_index_paths.get(name, {}),
        }
    if preconditioning is not None:
        if preconditioning.paths is None:
            raise ValueError("Preconditioning images need a source path")
        for name, path in preconditioning.views.items():
            index, _, _ = _index(
                path, project_root, kind="images", preconditioning=True
            )
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
            **preconditioning.model_dump(mode="json", exclude={"paths"}),
            "view_names": sorted(preconditioning.views),
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
            if source.path is None:
                raise ValueError(f"Image view {name!r} needs image_paths")
            if (
                source.signature is not None
                and _signature(source.path, project_root) != source.signature
            ):
                raise ValueError(f"Image view {name!r} changed after preprocessing")
        if preconditioning is not None and preconditioning.paths is None:
            raise ValueError("Preconditioning images need preconditioning_image_paths")
        if (
            preconditioning is not None
            and preconditioning.signature is not None
            and _preconditioning_signature(preconditioning, project_root)
            != preconditioning.signature
        ):
            raise ValueError("Preconditioning images changed after preprocessing")
        self.indices = {}
        self.frame_times: dict[str, dict[int, list[int]]] = {}
        self.video_index_paths: dict[Path, Path] = {}
        self.video_starts: dict[Path, int] = {}
        self.video_records: dict[Path, dict[str, Any]] = {}
        self.item_times: dict[int, list[int]] | None = None
        for name, source in sources.items():
            if source.path is None:
                raise ValueError(f"Image view {name!r} needs image_paths")
            index, timing, mode = _index(source.path, project_root, kind=source.type)
            self.indices[name] = index
            if source.type == "videos":
                if source.video_index_paths and set(source.video_index_paths) != set(
                    index
                ):
                    raise ValueError(
                        f"Image view {name!r} has incomplete video indexes"
                    )
                for sequence_id, files in index.items():
                    path = files[0]
                    if sequence_id in source.video_index_paths:
                        self.video_index_paths[path] = Path(
                            normalize_path(
                                source.video_index_paths[sequence_id], project_root
                            )
                        )
                    if source.timing_mode is not None:
                        self.video_starts[path] = _time_ms(
                            timing[sequence_id][0][0],
                            absolute=source.timing_mode == "absolute",
                        )
            if source.timing_mode is not None:
                if mode.rsplit(":", 1)[-1] != source.timing_mode:
                    raise ValueError(
                        f"Image view {name!r} timing changed after preprocessing"
                    )
                self.frame_times[name] = (
                    {}
                    if source.type == "videos"
                    else {
                        sequence_id: _source_frame_times(
                            source, files, timing[sequence_id], mode
                        )
                        for sequence_id, files in index.items()
                    }
                )
                if self.item_times is None:
                    if source.time_index_path is None:
                        raise ValueError(f"Image view {name!r} has no time index")
                    data = pl.read_parquet(
                        normalize_path(source.time_index_path, project_root)
                    )
                    self.item_times = {}
                    for sequence_id, frame in data.partition_by(
                        "sequenceId", as_dict=True
                    ).items():
                        key = (
                            sequence_id[0]
                            if isinstance(sequence_id, tuple)
                            else sequence_id
                        )
                        ordered = frame.sort("itemPosition")
                        if ordered["itemPosition"].to_list() != list(
                            range(ordered.height)
                        ):
                            raise ValueError(
                                f"Invalid item-time index for sequence {key}"
                            )
                        self.item_times[int(key)] = ordered["timeMs"].to_list()
                if (
                    source.time_index_signature is not None
                    and _item_times_signature(self.item_times)
                    != source.time_index_signature
                ):
                    raise ValueError("Item-time index changed after preprocessing")
        self.prefix_indices = (
            {
                name: _index(path, project_root, kind="images", preconditioning=True)[0]
                for name, path in sorted(preconditioning.views.items())
            }
            if preconditioning is not None
            else None
        )
        self.cache: OrderedDict[tuple[str, int, int, int], torch.Tensor] = OrderedDict()
        self.video_indices: dict[Path, _VideoIndex | None] = {}
        self.video_spools: dict[tuple[str, int, int], tuple[BinaryIO, int]] = {}

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
        self._remember_frame(key, self._transform(image, source))

    def _remember_frame(
        self, key: tuple[str, int, int, int], tensor: torch.Tensor
    ) -> None:
        self.cache[key] = tensor
        self.cache.move_to_end(key)
        if len(self.cache) > 128:
            self.cache.popitem(last=False)

    def _video_index(self, path: Path) -> _VideoIndex | None:
        if path not in self.video_indices:
            record = self._video_record(path)
            pts = tuple(record["pts"])
            keyframes = tuple(record["keyframes"])
            seek_points = tuple(record["seek_pts"])
            self.video_indices[path] = (
                _VideoIndex(
                    pts,
                    keyframes,
                    tuple(seek_points),
                    {stamp: i for i, stamp in enumerate(pts)},
                )
                if pts
                and all(stamp is not None for stamp in pts)
                and all(a < b for a, b in zip(pts, pts[1:]))
                and keyframes
                and keyframes[0] == 0
                and all(stamp is not None for stamp in seek_points)
                and len(keyframes) == len(seek_points)
                and len(set(pts)) == len(pts)
                else None
            )
        return self.video_indices[path]

    def _video_record(self, path: Path) -> dict[str, Any]:
        if path not in self.video_records:
            if path not in self.video_index_paths:
                self.video_records[path] = _scan_video(path)
            else:
                index_path = self.video_index_paths[path]
                try:
                    with index_path.open() as file:
                        record = json.load(file)
                    import av

                    with av.open(str(path)) as container:
                        base = container.streams.video[0].time_base
                        time_base = [base.numerator, base.denominator] if base else None
                    if (
                        not isinstance(record, dict)
                        or record.get("version") != _VIDEO_INDEX_VERSION
                        or record.get("media") != _media_signature(path)
                        or record.get("time_base") != time_base
                        or not isinstance(record.get("pts"), list)
                        or record.get("frame_count") != len(record["pts"])
                        or not isinstance(record.get("keyframes"), list)
                        or not isinstance(record.get("seek_pts"), list)
                        or any(
                            stamp is not None
                            and (not isinstance(stamp, int) or isinstance(stamp, bool))
                            for stamp in record["pts"]
                        )
                        or any(
                            stamp is not None
                            and (not isinstance(stamp, int) or isinstance(stamp, bool))
                            for stamp in record["seek_pts"]
                        )
                        or len(record["keyframes"]) != len(record["seek_pts"])
                        or (
                            record.get("frame_times_ms") is not None
                            and (
                                not isinstance(record["frame_times_ms"], list)
                                or len(record["frame_times_ms"]) != len(record["pts"])
                                or any(
                                    not isinstance(stamp, int)
                                    or isinstance(stamp, bool)
                                    for stamp in record["frame_times_ms"]
                                )
                            )
                        )
                        or any(
                            not isinstance(value, int) or isinstance(value, bool)
                            for value in record["keyframes"]
                        )
                        or any(
                            frame < 0 or frame >= len(record["pts"])
                            for frame in record["keyframes"]
                        )
                        or any(
                            a >= b
                            for a, b in zip(
                                record["keyframes"], record["keyframes"][1:]
                            )
                        )
                    ):
                        raise ValueError("invalid version, signature, or frame data")
                except (OSError, ValueError, TypeError, KeyError) as error:
                    raise ValueError(
                        f"Video index for {path} is missing, stale, or invalid: {error}"
                    ) from error
                self.video_records[path] = record
        return self.video_records[path]

    def _video_frame_times(self, name: str, sequence_id: int) -> list[int]:
        times = self.frame_times[name]
        if sequence_id not in times:
            path = self.indices[name][sequence_id][0]
            offsets = _validated_video_times(
                path, self._video_record(path)["frame_times_ms"]
            )
            times[sequence_id] = [
                self.video_starts[path] + offset for offset in offsets
            ]
        return times[sequence_id]

    def _spooled_video_frame(self, path: Path, frame: int, source: Any) -> torch.Tensor:
        import av
        import numpy as np

        key = (str(path), source.resize_size, source.crop_size)
        if key not in self.video_spools:
            spool = tempfile.TemporaryFile()
            try:
                with av.open(str(path)) as container:
                    count = 0
                    for decoded in container.decode(video=0):
                        image = self._transform(decoded.to_image(), source)
                        spool.write(image.contiguous().numpy().tobytes())
                        count += 1
                self.video_spools[key] = (spool, count)
            except Exception:
                spool.close()
                raise
        spool, count = self.video_spools[key]
        if frame >= count:
            raise ValueError(f"Missing decoded frame {frame} in {path}")
        width = 3 * source.crop_size * source.crop_size
        spool.seek(frame * width)
        data = spool.read(width)
        if len(data) != width:
            raise ValueError(f"Incomplete decoded frame {frame} in {path}")
        array = np.frombuffer(data, dtype=np.uint8).copy()
        return torch.from_numpy(array.reshape(3, source.crop_size, source.crop_size))

    def _prime_video(
        self, path: Path, frames: set[int], source: Any
    ) -> dict[int, torch.Tensor]:
        import av

        found = {
            frame: self.cache[(str(path), frame, source.resize_size, source.crop_size)]
            for frame in frames
            if (str(path), frame, source.resize_size, source.crop_size) in self.cache
        }
        for frame in found:
            self.cache.move_to_end(
                (str(path), frame, source.resize_size, source.crop_size)
            )
        missing = {frame for frame in frames if frame not in found}
        if not missing:
            return found
        index = self._video_index(path)
        if index is not None and max(missing) >= len(index.pts):
            raise ValueError(f"Missing decoded frames {sorted(missing)[:10]} in {path}")
        if index is not None:
            groups: dict[int, set[int]] = {}
            for frame in missing:
                anchor = index.keyframes[bisect_right(index.keyframes, frame) - 1]
                groups.setdefault(anchor, set()).add(frame)
            try:
                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    for anchor, requested in sorted(groups.items()):
                        pending = requested & missing
                        last = max(requested)
                        anchor_position = bisect_right(index.keyframes, anchor) - 1
                        container.seek(
                            index.seek_pts[anchor_position],
                            stream=stream,
                            backward=True,
                        )
                        for decoded in container.decode(video=0):
                            number = index.by_pts.get(decoded.pts)
                            if number in pending:
                                self._cache_frame(
                                    path, number, source, decoded.to_image()
                                )
                                found[number] = self.cache[
                                    (
                                        str(path),
                                        number,
                                        source.resize_size,
                                        source.crop_size,
                                    )
                                ]
                                missing.remove(number)
                                pending.remove(number)
                            if not pending or (number is not None and number >= last):
                                break
            except Exception:
                # A stream without reliable seeking still gets indexed reads.
                self.video_indices[path] = None
        if missing:
            self.video_indices[path] = None
            for frame in missing:
                key = (str(path), frame, source.resize_size, source.crop_size)
                tensor = self._spooled_video_frame(path, frame, source)
                self._remember_frame(key, tensor)
                found[frame] = tensor
            missing.clear()
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
            capacity = source.frame_capacity
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
                    if source.timing_mode is None:
                        selected = source.alignment.frame_indices(item)[-capacity:]
                    else:
                        assert self.item_times is not None
                        item_times = self.item_times[sequence_id]
                        if item >= len(item_times):
                            raise ValueError(
                                f"Missing item time for sequence {sequence_id}, item {item}"
                            )
                        frame_times = (
                            self._video_frame_times(name, sequence_id)
                            if source.type == "videos"
                            else self.frame_times[name][sequence_id]
                        )
                        stop = bisect_right(frame_times, item_times[item])
                        previous = (
                            bisect_right(frame_times, item_times[item - 1])
                            if item > 0
                            else 0
                        )
                        selected = range(max(previous, stop - capacity), stop)
                        if not selected and stop:
                            selected = range(stop - 1, stop)
                    for slot, frame in enumerate(selected):
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
