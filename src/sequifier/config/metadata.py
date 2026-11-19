"""Typed preprocessing metadata used during config resolution."""

from __future__ import annotations

import copy
import json
from typing import Any, Literal

from pydantic import ConfigDict, Field, ValidationInfo, field_validator, model_validator

from sequifier.config.depth_layout import DepthLayoutRegistryModel
from sequifier.config.image_config import ImageSourceConfig, PreconditioningImageConfig
from sequifier.config.naming import ConfigModel
from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.helpers import ModelWindowView, StoredWindowLayout
from sequifier.special_tokens import SPECIAL_TOKEN_IDS, validate_special_token_ids
from sequifier.typechecking import beartype

RESOLVED_ONLY_CONFIG_KEYS = {
    "categorical_columns",
    "real_columns",
    "id_maps",
    "cardinality_config",
    "special_token_ids",
    "storage_layout",
    "window_view",
    "n_classes",
    "window_length",
    "window_lengths",
    "max_target_offset",
    "stored_window_layout_version",
    "depth_layouts",
    "image_sources",
    "time_alignment_column",
    "preconditioning_image",
    "tensor_payload_kind",
    "tensor_payload_version",
    "prediction_aligned_splits",
    "split_index",
}


TensorPayloadKind = Literal["tabular", "depth"]

_TENSOR_PAYLOAD_VERSIONS: dict[TensorPayloadKind, int] = {
    "tabular": ContractVersions.DATASET_TENSOR_METADATA,
    "depth": ContractVersions.DEPTH_TENSOR_METADATA,
}


def dataset_tensor_kind(depth_layouts) -> TensorPayloadKind:
    return "depth" if depth_layouts else "tabular"


def dataset_tensor_version(kind: TensorPayloadKind) -> int:
    return _TENSOR_PAYLOAD_VERSIONS[kind]


def validate_tensor_payload_contract(kind, version, depth_layouts, *, projected=False):
    expected_kind = dataset_tensor_kind(depth_layouts)
    # A shallow model interface may come from a dataset that has depth columns.
    allowed = (
        ("tabular", "depth") if projected and not depth_layouts else (expected_kind,)
    )
    if kind not in allowed:
        raise ValueError(
            f"tensor_payload_kind must be one of {allowed!r}, got {kind!r}"
        )
    require_contract_version(
        f"{kind} dataset tensor metadata", version, dataset_tensor_version(kind)
    )


def validate_persisted_metadata_versions(values: dict) -> None:
    if not isinstance(values, dict):
        raise ValueError("Dataset metadata must contain a mapping")
    validate_tensor_payload_contract(
        values.get("tensor_payload_kind"),
        values.get("tensor_payload_version"),
        values.get("depth_layouts"),
    )
    require_contract_version(
        "stored-window layout",
        values.get("stored_window_layout_version"),
        ContractVersions.STORED_WINDOW_LAYOUT,
    )


class DatasetMetadata(ConfigModel):
    """The stable subset of preprocessing metadata consumed by other commands."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_inline_metadata(cls, values: Any) -> Any:
        if isinstance(values, dict) and {
            "window_strides",
            "window_placement",
            "split_context",
        } & set(values):
            raise ValueError(
                "Preprocessing metadata uses obsolete placement settings; re-run preprocessing"
            )
        if isinstance(values, dict):
            kind = dataset_tensor_kind(values.get("depth_layouts"))
            values = {
                "tensor_payload_kind": kind,
                "tensor_payload_version": dataset_tensor_version(kind),
                **values,
            }
        return values

    depth_layouts: DepthLayoutRegistryModel = Field(
        default_factory=DepthLayoutRegistryModel
    )
    tensor_payload_kind: TensorPayloadKind
    tensor_payload_version: int = Field(strict=True)
    split_paths: list[str] = Field(default_factory=list)
    split_index: int | None = Field(default=None, ge=0)
    column_data_types: dict[str, str] = Field(
        default_factory=dict,
    )
    static_columns: list[str] = Field(default_factory=list)
    static_features_path: str | None = None
    image_sources: dict[str, ImageSourceConfig] = Field(default_factory=dict)
    time_alignment_column: str | None = None
    preconditioning_image: PreconditioningImageConfig | None = None
    n_classes: dict[str, int] = Field(default_factory=dict)
    id_maps: dict[str, dict[str | int, int]] = Field(default_factory=dict)
    cardinality_config: dict[str, dict[str, Any]] = Field(default_factory=dict)
    special_token_ids: dict[str, int] = Field(
        default_factory=lambda: dict(SPECIAL_TOKEN_IDS.ids_by_label)
    )
    selected_columns_statistics: dict[str, dict[str, float]] = Field(
        default_factory=dict
    )
    normalize_real_columns: bool = True
    normalize_on_all_data: bool = False
    prediction_aligned_splits: list[int] = Field(default_factory=list)
    prediction_alignment_steps: int = Field(default=1, gt=0)
    prediction_length: int | None = Field(default=None, gt=0)
    target_offset: int | None = Field(default=None, ge=0)
    window_length: int = Field(gt=0)
    window_lengths: list[int] | None = None
    max_target_offset: int = Field(default=1, ge=0)
    stored_window_layout_version: int = Field(
        default=ContractVersions.STORED_WINDOW_LAYOUT,
        strict=True,
        json_schema_extra={"const": ContractVersions.STORED_WINDOW_LAYOUT},
    )

    @model_validator(mode="after")
    def validate_layout_metadata(self, info: ValidationInfo):
        validate_tensor_payload_contract(
            self.tensor_payload_kind,
            self.tensor_payload_version,
            self.depth_layouts,
            projected=(info.context or {}).get("projected_depth_layouts", False),
        )
        require_contract_version(
            "stored-window layout",
            self.stored_window_layout_version,
            ContractVersions.STORED_WINDOW_LAYOUT,
        )
        if self.window_lengths is not None and (
            (self.split_paths and len(self.window_lengths) != len(self.split_paths))
            or (
                self.split_index is not None
                and self.split_index >= len(self.window_lengths)
            )
            or any(width <= self.max_target_offset for width in self.window_lengths)
        ):
            raise ValueError(
                "Metadata window_lengths must contain one valid width per split"
            )
        if self.window_lengths is not None:
            expected_width = self.width_for_split(self.split_index or 0)
            if self.window_length != expected_width:
                raise ValueError(
                    "Metadata window_length does not match selected split width"
                )
        if self.prediction_aligned_splits:
            if self.prediction_length is None or self.target_offset is None:
                raise ValueError(
                    "Aligned metadata requires prediction_length and target_offset"
                )
            if self.target_offset != self.max_target_offset:
                raise ValueError(
                    "Aligned metadata requires target_offset == max_target_offset"
                )
            if len(self.prediction_aligned_splits) != len(
                set(self.prediction_aligned_splits)
            ) or any(
                isinstance(i, bool)
                or i < 0
                or (self.split_paths and i >= len(self.split_paths))
                or (self.window_lengths is not None and i >= len(self.window_lengths))
                for i in self.prediction_aligned_splits
            ):
                raise ValueError("Aligned metadata has invalid split indices")
            if any(
                self.prediction_alignment_steps * self.prediction_length
                > self.width_for_split(i) - self.max_target_offset
                for i in self.prediction_aligned_splits
            ):
                raise ValueError(
                    "Aligned metadata window width cannot hold all prediction steps"
                )
        if (
            self.split_index is not None
            and self.split_paths
            and self.split_index >= len(self.split_paths)
        ):
            raise ValueError("Metadata split_index is outside split_paths")
        return self

    def width_for_split(self, index: int) -> int:
        return (
            self.window_lengths[index]
            if self.window_lengths is not None
            else self.window_length
        )

    def layout_for_split(self, index: int) -> StoredWindowLayout:
        return StoredWindowLayout(
            self.width_for_split(index),
            self.max_target_offset,
            self.stored_window_layout_version,
        )

    @field_validator("special_token_ids")
    @classmethod
    @beartype
    def validate_token_ids(cls, value: dict[str, int]) -> dict[str, int]:
        return validate_special_token_ids(value, source="dataset metadata")

    @property
    @beartype
    def storage_layout(self) -> StoredWindowLayout:
        return StoredWindowLayout(
            window_length=self.window_length,
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
    validate_persisted_metadata_versions(values)
    return DatasetMetadata.model_validate(values)


@beartype
def extract_inline_metadata(
    values: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Separate inline metadata facts from the authored request."""

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

    tensor_kind = dataset_tensor_kind(authored.get("depth_layouts"))
    metadata_values: dict[str, Any] = {
        "split_paths": [
            path
            for path in (
                authored.get("data_path"),
                authored.get("validation_data_path"),
            )
            if path is not None
        ],
        "depth_layouts": authored.get("depth_layouts", {}),
        "tensor_payload_kind": authored.get("tensor_payload_kind", tensor_kind),
        "tensor_payload_version": authored.get(
            "tensor_payload_version", dataset_tensor_version(tensor_kind)
        ),
        "column_data_types": authored.get("column_data_types", {}),
        "static_columns": authored.get("static_columns", []),
        "static_features_path": authored.get("static_features_path"),
        "image_sources": authored.get("image_sources", {}),
        "time_alignment_column": authored.get("time_alignment_column"),
        "preconditioning_image": authored.get("preconditioning_image"),
        "n_classes": authored.get("n_classes", {}),
        "id_maps": authored.get("id_maps", {}),
        "cardinality_config": authored.get("cardinality_config", {}),
        "special_token_ids": authored.get(
            "special_token_ids", SPECIAL_TOKEN_IDS.ids_by_label
        ),
        "selected_columns_statistics": authored.get("selected_columns_statistics", {}),
        "normalize_real_columns": authored.get("normalize_real_columns", True),
        "normalize_on_all_data": authored.get("normalize_on_all_data", False),
        "prediction_aligned_splits": authored.get("prediction_aligned_splits", []),
        "prediction_alignment_steps": authored.get("prediction_alignment_steps", 1),
        "window_lengths": authored.get("window_lengths"),
        "split_index": authored.get("split_index"),
        "prediction_length": authored.get("prediction_length"),
        "target_offset": authored.get("target_offset"),
    }

    storage_layout = authored.get("storage_layout")
    if isinstance(storage_layout, StoredWindowLayout):
        metadata_values.update(
            {
                "window_length": storage_layout.window_length,
                "max_target_offset": storage_layout.max_target_offset,
                "stored_window_layout_version": storage_layout.version,
            }
        )
    elif isinstance(storage_layout, dict):
        metadata_values.update(
            {
                "window_length": storage_layout.get("window_length"),
                "max_target_offset": storage_layout.get("max_target_offset", 1),
                "stored_window_layout_version": storage_layout.get(
                    "version", ContractVersions.STORED_WINDOW_LAYOUT
                ),
            }
        )
    else:
        metadata_values.update(
            {
                "window_length": authored.get("window_length"),
                "max_target_offset": authored.get("max_target_offset", 1),
                "stored_window_layout_version": authored.get(
                    "stored_window_layout_version",
                    ContractVersions.STORED_WINDOW_LAYOUT,
                ),
            }
        )

    for key in RESOLVED_ONLY_CONFIG_KEYS:
        authored.pop(key, None)
    for key in (
        "selected_columns_statistics",
        "normalize_real_columns",
        "normalize_on_all_data",
        "prediction_aligned_splits",
    ):
        authored.pop(key, None)
    if metadata_values["window_length"] is None:
        return authored, None
    return authored, metadata_values
