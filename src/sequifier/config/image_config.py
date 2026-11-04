"""Image source and frame-alignment configuration."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrameAlignment(BaseModel):
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


class ImageSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    type: Literal["images", "videos"] = "images"
    alignment: FrameAlignment = Field(default_factory=FrameAlignment)
    max_frames_per_item: int | None = Field(default=None, gt=0)
    resize_size: int = Field(default=256, gt=0)
    crop_size: int = Field(default=224, gt=0)
    signature: str | None = None
    timing_mode: Literal["relative", "absolute"] | None = None
    time_index_path: str | None = None
    time_index_signature: str | None = None

    @property
    def frame_capacity(self) -> int:
        return self.max_frames_per_item or self.alignment.max_frames_per_item

    @model_validator(mode="after")
    def validate_size(self):
        if self.crop_size > self.resize_size:
            raise ValueError("crop_size must not exceed resize_size")
        return self


class PreconditioningImageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str | dict[str, str]
    resize_size: int = Field(default=256, gt=0)
    crop_size: int = Field(default=224, gt=0)
    signature: str | None = None

    @model_validator(mode="after")
    def validate_size(self):
        if self.crop_size > self.resize_size:
            raise ValueError("crop_size must not exceed resize_size")
        if isinstance(self.path, dict) and (
            not self.path
            or any(
                not name.isidentifier() or not path for name, path in self.path.items()
            )
        ):
            raise ValueError("Preconditioning views require names and folder paths")
        if isinstance(self.path, str) and not self.path:
            raise ValueError("Preconditioning image path cannot be empty")
        return self

    @property
    def views(self) -> dict[str, str]:
        return self.path if isinstance(self.path, dict) else {"default": self.path}

    def compatibility_signature(self) -> dict:
        return {
            **self.model_dump(exclude={"path", "signature"}),
            "views": sorted(self.views),
        }


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
