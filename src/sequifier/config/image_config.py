"""Image source and frame-alignment configuration."""

from typing import Any, Literal, Optional

from pydantic import ConfigDict, Field, model_validator

from sequifier.config.naming import ConfigModel


class FrameAlignment(ConfigModel):
    model_config = ConfigDict(extra="forbid")

    frames: int = Field(default=1, gt=0)
    items: int = Field(default=1, gt=0)
    phase: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_phase(self):
        if self.phase >= self.items:
            raise ValueError("alignment.phase must be smaller than alignment.items")
        return self

    def boundary(self, item: int) -> int:
        """First frame assigned to an item when frames arrive at least as fast."""
        numerator = item * self.frames + self.phase
        return (numerator + self.items - 1) // self.items - (
            self.phase + self.items - 1
        ) // self.items

    def frame_at(self, item: int) -> int:
        """Latest frame available at an item when frames arrive more slowly."""
        return (item * self.frames + self.phase) // self.items

    def frame_indices(self, item: int) -> range:
        if self.frames >= self.items:
            return range(self.boundary(item), self.boundary(item + 1))
        return range(self.frame_at(item), self.frame_at(item) + 1)

    def expected_frames(self, item_count: int) -> int:
        if item_count <= 0:
            return 0
        if self.frames >= self.items:
            return self.boundary(item_count)
        return self.frame_at(item_count - 1) + 1

    @property
    def max_frames_per_item(self) -> int:
        return (self.frames + self.items - 1) // self.items


class ImageSourceConfig(ConfigModel):
    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    type: Literal["images", "videos"] = "images"
    alignment: FrameAlignment = Field(default_factory=FrameAlignment)
    max_frames_per_item: int | None = Field(default=None, gt=0)
    resize_size: int = Field(default=256, gt=0)
    crop_size: int = Field(default=224, gt=0)
    signature: str | None = None
    timing_mode: Literal["relative", "absolute"] | None = None
    time_index_path: str | None = None
    time_index_signature: str | None = None
    video_index_paths: dict[int, str] = Field(default_factory=dict)

    @property
    def frame_capacity(self) -> int:
        return self.max_frames_per_item or self.alignment.max_frames_per_item

    @model_validator(mode="after")
    def validate_size(self):
        if self.crop_size > self.resize_size:
            raise ValueError("crop_size must not exceed resize_size")
        if (
            self.max_frames_per_item is not None
            and self.max_frames_per_item < self.alignment.max_frames_per_item
        ):
            raise ValueError(
                "max_frames_per_item cannot be smaller than the alignment frame count"
            )
        return self


class PreconditioningImageConfig(ConfigModel):
    model_config = ConfigDict(extra="forbid")

    paths: dict[str, str] | None = None
    view_names: list[str] = Field(default_factory=list)
    resize_size: int = Field(default=256, gt=0)
    crop_size: int = Field(default=224, gt=0)
    signature: str | None = None

    @model_validator(mode="after")
    def validate_size(self):
        if self.crop_size > self.resize_size:
            raise ValueError("crop_size must not exceed resize_size")
        if self.paths is not None and (
            not self.paths
            or any(
                not name.isidentifier() or not path for name, path in self.paths.items()
            )
        ):
            raise ValueError("Preconditioning views require names and folder paths")
        if self.paths is None and not self.view_names:
            raise ValueError("Preconditioning image views are required")
        return self

    @property
    def views(self) -> dict[str, str]:
        return (
            self.paths
            if self.paths is not None
            else {name: "" for name in self.view_names}
        )

    def compatibility_signature(self) -> dict:
        return {
            **self.model_dump(exclude={"paths", "signature", "view_names"}),
            "views": sorted(self.views),
        }


MediaPath = Optional[dict[str, str]]


def bind_image_paths(
    sources: dict[str, ImageSourceConfig],
    preconditioning: PreconditioningImageConfig | None,
    image_paths: MediaPath,
    preconditioning_image_paths: MediaPath,
) -> tuple[dict[str, ImageSourceConfig], PreconditioningImageConfig | None]:
    """Attach explicit named media locations to path-free metadata."""
    paths = image_paths or {}
    if set(paths) != set(sources) or any(not path for path in paths.values()):
        raise ValueError("image_paths must provide exactly the selected image views")
    bound = {
        name: source.model_copy(update={"path": paths[name]})
        for name, source in sources.items()
    }
    if preconditioning is None:
        if preconditioning_image_paths is not None:
            raise ValueError("preconditioning_image_paths has no metadata source")
        return bound, None
    prefix_paths = preconditioning_image_paths or {}
    if set(prefix_paths) != set(preconditioning.views) or any(
        not path for path in prefix_paths.values()
    ):
        raise ValueError(
            "preconditioning_image_paths must provide exactly the selected views"
        )
    return bound, preconditioning.model_copy(update={"paths": prefix_paths})


def image_views(ingestion: Any) -> set[str]:
    """Names of temporal image views consumed by an ingestion tree."""
    if ingestion.type == "images":
        return set(ingestion.views)
    if ingestion.type == "structured":
        return set(ingestion.image_cells.values())
    if ingestion.type == "composite":
        return set().union(
            *(image_views(child) for child in ingestion.branches.values())
        )
    return set()
