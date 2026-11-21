"""Typed preprocessing metadata used during config resolution."""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import (
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_serializer,
    model_validator,
)

from sequifier.config.depth_layout import DepthLayoutRegistryModel
from sequifier.config.image_config import ImageSourceConfig, PreconditioningImageConfig
from sequifier.config.naming import ConfigModel
from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.helpers import (
    StoredWindowLayout,
    canonicalize_polars_dtype_name,
    is_integer_dtype_name,
)
from sequifier.special_tokens import SPECIAL_TOKEN_IDS, validate_special_token_ids
from sequifier.typechecking import beartype

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


class FeatureTransforms(ConfigModel):
    """Fitted categorical and numeric transforms for model inputs."""

    column_data_types: dict[str, str] = Field(default_factory=dict)
    n_classes: dict[str, int] = Field(default_factory=dict)
    id_maps: dict[str, dict[str | int, int]] = Field(default_factory=dict)
    selected_columns_statistics: dict[str, dict[str, float]] = Field(
        default_factory=dict
    )
    normalize_real_columns: bool = True


class FeatureMetadata(FeatureTransforms):
    special_token_ids: dict[str, int] = Field(
        default_factory=lambda: dict(SPECIAL_TOKEN_IDS.ids_by_label)
    )
    depth_layouts: DepthLayoutRegistryModel = Field(
        default_factory=DepthLayoutRegistryModel
    )
    image_sources: dict[str, ImageSourceConfig] = Field(default_factory=dict)
    preconditioning_image: PreconditioningImageConfig | None = None

    def split_columns(self, columns: Collection[str]) -> tuple[list[str], list[str]]:
        categorical, real = [], []
        for name in columns:
            if is_integer_dtype_name(self.column_data_types[name]):
                categorical.append(name)
            else:
                real.append(name)
        return categorical, real

    def feature_values(self):
        return {key: getattr(self, key) for key in FeatureMetadata.model_fields}

    def signature(self):
        values = self.feature_values()
        values["depth_layouts"] = self.depth_layouts.compatibility_signature(
            self.column_data_types
        )
        values["image_sources"] = {
            name: image.compatibility_signature()
            for name, image in self.image_sources.items()
        }
        if self.preconditioning_image:
            values["preconditioning_image"] = (
                self.preconditioning_image.compatibility_signature()
            )
        return values

    def select_media(self, names: Collection[str], use_prefix: bool) -> dict[str, Any]:
        if missing := set(names) - self.image_sources.keys():
            raise ValueError(f"Metadata lacks image views: {sorted(missing)}")
        if use_prefix and self.preconditioning_image is None:
            raise ValueError(
                "Metadata lacks the configured preconditioning image source"
            )
        return {
            "image_sources": {name: self.image_sources[name] for name in sorted(names)},
            "preconditioning_image": self.preconditioning_image if use_prefix else None,
        }

    def select_features(
        self,
        columns: Collection[str],
        image_names: Collection[str],
        use_prefix: bool,
        *,
        model_inputs: Collection[str] | None = None,
    ) -> FeatureMetadata:
        def selected(mapping, names=columns):
            return {name: mapping[name] for name in names if name in mapping}

        categorical, real = (
            (columns, columns) if model_inputs is None else self.split_columns(columns)
        )

        return FeatureMetadata(
            column_data_types={
                name: dtype
                if dtype.startswith(("Date", "Time", "Duration"))
                else canonicalize_polars_dtype_name(dtype)
                for name, dtype in selected(self.column_data_types).items()
            },
            id_maps=selected(self.id_maps, categorical),
            n_classes=selected(self.n_classes, categorical),
            selected_columns_statistics=selected(
                self.selected_columns_statistics, real
            ),
            special_token_ids=self.special_token_ids,
            normalize_real_columns=self.normalize_real_columns,
            depth_layouts=self.depth_layouts.relevant_layouts(
                columns if model_inputs is None else model_inputs
            ),
            **self.select_media(image_names, use_prefix),
        )


@dataclass(frozen=True)
class FeatureSelection:
    """Ordered model roles and media selected from a preprocessing contract."""

    inputs: list[str]
    targets: list[str]
    static: list[str]
    images: set[str]
    use_prefix: bool

    @property
    def columns(self) -> list[str]:
        return list(dict.fromkeys([*self.inputs, *self.targets, *self.static]))

    def resolve(self, metadata: DatasetMetadata) -> FeatureMetadata:
        columns = self.columns
        if missing := set(columns) - metadata.column_data_types.keys():
            raise ValueError(f"Metadata lacks selected columns: {sorted(missing)}")
        if missing := set(self.static) - set(metadata.static_columns):
            raise ValueError(f"Metadata lacks static features: {sorted(missing)}")
        if any(metadata.depth_layouts.is_deep_column(name) for name in self.targets):
            raise ValueError("Depth targets are not supported")
        if image_targets := set(self.targets).intersection(self.images):
            raise ValueError(
                f"Image views cannot be target columns: {sorted(image_targets)}"
            )

        return metadata.select_features(
            columns, self.images, self.use_prefix, model_inputs=self.inputs
        )


class RawInputMetadata(ConfigModel):
    cardinality_config: dict[str, dict[str, Any]] = Field(default_factory=dict)
    raw_mask_column: str | None = None
    time_alignment_column: str | None = None

    def raw_input_contract(self, columns):
        return {
            "cardinality_config": {
                name: self.cardinality_config[name]
                for name in columns
                if name in self.cardinality_config
            },
            "raw_mask_column": self.raw_mask_column,
            "time_alignment_column": self.time_alignment_column,
        }


class DatasetMetadata(FeatureMetadata, RawInputMetadata):
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

    tensor_payload_kind: TensorPayloadKind
    tensor_payload_version: int = Field(strict=True)
    split_names: list[str] = Field(default_factory=list)
    split_paths: list[str] = Field(default_factory=list)
    split_index: int | None = Field(default=None, ge=0)
    static_columns: list[str] = Field(default_factory=list)
    static_features_path: str | None = None
    indexed_views: list[str] = Field(default_factory=list)
    indexed_inline_media: bool = False
    indexed_derived_features: list[str] = Field(default_factory=list)
    indexed_passthrough: list[str] = Field(default_factory=list)
    fit_split_indices: list[int] = Field(default_factory=list)
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

    def named_split_index(self, name):
        names = self.split_names
        if name not in names:
            raise ValueError(
                f"Unknown metadata split {name!r}; available splits are {names}"
            )
        return names.index(name)

    @model_validator(mode="after")
    def validate_layout_metadata(self, info: ValidationInfo):
        if len(self.split_names) != len(set(self.split_names)) or len(
            self.split_names
        ) != len(self.split_paths):
            raise ValueError("Metadata must provide one distinct split name per path")
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
        if self.indexed_views:
            expected = [str(index) for index in range(len(self.split_paths))]
            if self.indexed_views not in (expected, ["predict"]) or (
                len(self.indexed_views) != len(self.split_paths)
            ):
                raise ValueError("Indexed metadata requires one named view per split")
            if set(self.fit_split_indices) - set(range(len(self.split_paths))):
                raise ValueError("Indexed metadata names an unknown fitting split")
            if set(self.indexed_derived_features) - self.column_data_types.keys():
                raise ValueError("Indexed derived features require declared dtypes")
        return self

    @model_serializer(mode="wrap")
    def serialize_metadata(self, handler):
        values = handler(self)
        if not self.indexed_views:
            for field in (
                "indexed_views",
                "indexed_inline_media",
                "indexed_derived_features",
                "indexed_passthrough",
                "fit_split_indices",
            ):
                values.pop(field, None)
        return values

    def feature_contract(self, columns, image_names, use_prefix, *, raw_inputs=False):
        contract = self.select_features(columns, image_names, use_prefix).signature()
        if raw_inputs:
            contract.update(self.raw_input_contract(columns))
        return contract

    def window_contract(self):
        return {
            "storage_layout": self.storage_layout,
            "prediction_aligned_splits": self.prediction_aligned_splits,
            "prediction_alignment_steps": self.prediction_alignment_steps,
            "window_lengths": self.window_lengths,
            "prediction_length": self.prediction_length,
            "target_offset": self.target_offset,
        }

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
    if values.get("indexed_views"):
        from pathlib import Path

        root = Path(path).absolute().parent
        values["split_paths"] = [str(root / name) for name in values["split_paths"]]
        if values.get("static_features_path"):
            values["static_features_path"] = str(root / values["static_features_path"])
        for source in values.get("image_sources", {}).values():
            if source.get("path"):
                source["path"] = str(root / source["path"])
            if source.get("time_index_path"):
                source["time_index_path"] = str(root / source["time_index_path"])
            paths = source.get("video_index_paths", {})
            source["video_index_paths"] = {
                sequence: str(root / name) for sequence, name in paths.items()
            }
        prefix = values.get("preconditioning_image")
        if prefix and prefix.get("paths"):
            prefix["paths"] = {
                name: str(root / location) for name, location in prefix["paths"].items()
            }
    return DatasetMetadata.model_validate(values)


def validate_authored_metadata(values: dict) -> None:
    layout = values.get("storage_layout")
    if (
        "tensor_payload_version" in values
        or "stored_window_layout_version" in values
        or (isinstance(layout, dict) and "version" in layout)
    ):
        raise ValueError("Inline metadata cannot declare internal storage versions")
