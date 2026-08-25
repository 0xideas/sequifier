"""Typed preprocessing metadata used during config resolution."""

from __future__ import annotations

import copy
import json
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from sequifier.helpers import ModelWindowView, StoredWindowLayout
from sequifier.special_tokens import SPECIAL_TOKEN_IDS, validate_special_token_ids
from sequifier.typechecking import beartype

RESOLVED_ONLY_CONFIG_KEYS = {
    "categorical_columns",
    "real_columns",
    "id_maps",
    "special_token_ids",
    "storage_layout",
    "window_view",
    "n_classes",
    "stored_context_width",
    "max_target_offset",
    "stored_window_layout_version",
}


class DatasetMetadata(BaseModel):
    """The stable subset of preprocessing metadata consumed by other commands."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    split_paths: list[str] = Field(default_factory=list)
    column_data_types: dict[str, str] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("column_data_types", "column_types"),
    )
    n_classes: dict[str, int] = Field(default_factory=dict)
    id_maps: dict[str, dict[str | int, int]] = Field(default_factory=dict)
    special_token_ids: dict[str, int] = Field(
        default_factory=lambda: dict(SPECIAL_TOKEN_IDS.ids_by_label)
    )
    selected_columns_statistics: dict[str, dict[str, float]] = Field(
        default_factory=dict
    )
    normalize_real_columns: bool = True
    stored_context_width: int = Field(gt=0)
    max_target_offset: int = Field(default=1, ge=0)
    stored_window_layout_version: int = 2

    @field_validator("special_token_ids")
    @classmethod
    @beartype
    def validate_token_ids(cls, value: dict[str, int]) -> dict[str, int]:
        return validate_special_token_ids(value, source="dataset metadata")

    @property
    @beartype
    def storage_layout(self) -> StoredWindowLayout:
        return StoredWindowLayout(
            stored_context_width=self.stored_context_width,
            max_target_offset=self.max_target_offset,
            version=self.stored_window_layout_version,
        )


@beartype
def load_dataset_metadata(path: str) -> DatasetMetadata:
    """Load and validate one preprocessing metadata JSON file."""

    with open(path, "r") as file:
        values: Any = json.load(file)
    if not isinstance(values, dict):
        raise ValueError(f"Metadata config '{path}' must contain a JSON object.")
    return DatasetMetadata.model_validate(values)


@beartype
def extract_inline_metadata(
    values: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Split the historical ``skip_metadata`` representation into two mappings."""

    authored = copy.deepcopy(values)
    window_view = authored.get("window_view")
    if "context_length" not in authored and isinstance(window_view, dict):
        authored["context_length"] = window_view.get("context_length")
    elif "context_length" not in authored and isinstance(window_view, ModelWindowView):
        authored["context_length"] = window_view.context_length
    if "target_offset" not in authored and isinstance(window_view, dict):
        authored["target_offset"] = window_view.get("target_offset", 1)
    elif "target_offset" not in authored and isinstance(window_view, ModelWindowView):
        authored["target_offset"] = window_view.target_offset

    metadata_values: dict[str, Any] = {
        "split_paths": [
            path
            for path in (
                authored.get("data_path"),
                authored.get("validation_data_path"),
            )
            if path is not None
        ],
        "column_data_types": authored.get("column_data_types", {}),
        "n_classes": authored.get("n_classes", {}),
        "id_maps": authored.get("id_maps", {}),
        "special_token_ids": authored.get(
            "special_token_ids", SPECIAL_TOKEN_IDS.ids_by_label
        ),
        "selected_columns_statistics": authored.get("selected_columns_statistics", {}),
        "normalize_real_columns": authored.get("normalize_real_columns", True),
    }

    storage_layout = authored.get("storage_layout")
    if isinstance(storage_layout, StoredWindowLayout):
        metadata_values.update(
            {
                "stored_context_width": storage_layout.stored_context_width,
                "max_target_offset": storage_layout.max_target_offset,
                "stored_window_layout_version": storage_layout.version,
            }
        )
    elif isinstance(storage_layout, dict):
        metadata_values.update(
            {
                "stored_context_width": storage_layout.get("stored_context_width"),
                "max_target_offset": storage_layout.get("max_target_offset", 1),
                "stored_window_layout_version": storage_layout.get("version", 2),
            }
        )
    else:
        metadata_values.update(
            {
                "stored_context_width": authored.get("stored_context_width"),
                "max_target_offset": authored.get("max_target_offset", 1),
                "stored_window_layout_version": authored.get(
                    "stored_window_layout_version", 2
                ),
            }
        )

    for key in RESOLVED_ONLY_CONFIG_KEYS:
        authored.pop(key, None)
    for key in ("selected_columns_statistics", "normalize_real_columns"):
        authored.pop(key, None)
    if metadata_values["stored_context_width"] is None:
        return authored, None
    return authored, metadata_values
