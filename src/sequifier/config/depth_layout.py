"""Portable named child-collection layouts and selected-interface signatures."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
from torch import nn


def depth_mask_metadata_key(layout: str) -> str:
    return f"depth_valid_mask:{layout}"


class DepthLayoutModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position_column: str
    columns: list[str] = Field(min_length=1)
    context_length: int = Field(gt=0)
    position_base: Literal[0, 1] = 0
    allow_gaps: bool = False

    @model_validator(mode="after")
    def validate_columns(self):
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Depth layout columns must be unique")
        if set(self.columns) & {"sequenceId", "itemPosition"}:
            raise ValueError("Outer coordinates cannot be depth features")
        if self.position_column in {"sequenceId", "itemPosition", *self.columns}:
            raise ValueError(
                "Depth position columns cannot be features or outer coordinates"
            )
        return self


class DepthLayoutRegistryModel(RootModel[dict[str, DepthLayoutModel]]):
    root: dict[str, DepthLayoutModel] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_registry(self):
        occupied = set()
        storage = nn.ModuleDict()
        for name, layout in self.root.items():
            if not name or "." in name or hasattr(storage, name):
                raise ValueError(f"Invalid depth layout module key: {name!r}")
            if occupied.intersection(layout.columns):
                raise ValueError("A feature cannot belong to multiple depth layouts")
            occupied.update(layout.columns)
        if occupied.intersection(x.position_column for x in self.root.values()):
            raise ValueError("Depth position columns cannot be layout features")
        return self

    def __bool__(self):
        return bool(self.root)

    def __contains__(self, name):
        return name in self.root

    def __getitem__(self, name):
        return self.root[name]

    def items(self):
        return self.root.items()

    @property
    def deep_columns(self) -> tuple[str, ...]:
        """Lexically ordered feature names; never used for concatenation order."""
        return tuple(sorted(self.column_to_layout))

    @property
    def column_to_layout(self) -> dict[str, str]:
        return {
            column: name
            for name in sorted(self.root)
            for column in self.root[name].columns
        }

    def is_deep_column(self, column: str) -> bool:
        return column in self.column_to_layout

    def layout_for_column(self, column: str) -> DepthLayoutModel | None:
        name = self.column_to_layout.get(column)
        return self.root[name] if name is not None else None

    def relevant_layouts(self, columns) -> "DepthLayoutRegistryModel":
        selected = set(columns)
        return type(self)(
            {
                name: layout.model_copy(
                    update={"columns": [c for c in layout.columns if c in selected]}
                )
                for name, layout in sorted(self.root.items())
                if selected.intersection(layout.columns)
            }
        )

    def compatibility_signature(self, columns) -> dict:
        return {
            name: {**layout.model_dump(), "columns": sorted(layout.columns)}
            for name, layout in self.relevant_layouts(columns).items()
        }
