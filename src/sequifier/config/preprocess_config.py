import os
import warnings
from datetime import date, datetime
from typing import Any, Optional, Union

import numpy as np
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from sequifier.config.composition import (
    load_composed_yaml_config,
    merge_config_fragments,
)
from sequifier.config.depth_layout import DepthLayoutRegistryModel
from sequifier.config.split_context import SplitContextConfig
from sequifier.helpers import (
    canonicalize_polars_dtype_name,
    is_float_dtype_name,
    is_integer_dtype_name,
    try_catch_excess_keys,
)
from sequifier.typechecking import beartype


@beartype
def load_preprocessor_config(
    config_path: str, args_config: dict
) -> "PreprocessorModel":
    """Load preprocessing YAML plus CLI overrides."""
    config_values = load_composed_yaml_config(config_path)

    config_values = merge_config_fragments((config_values, args_config))

    return try_catch_excess_keys(config_path, PreprocessorModel, config_values)


class CardinalityHashingModel(BaseModel):
    """Hash non-retained categorical values into a bounded set of buckets."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    num_buckets: int = Field(
        ge=1,
        validation_alias=AliasChoices("num_buckets", "buckets"),
    )
    seed: int = 0


class CardinalityLimitModel(BaseModel):
    """Bound the vocabulary produced for one categorical column."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    min_freq: Optional[int] = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("min_freq", "min_count"),
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("top_k", "k", "max_categories"),
    )
    hashing: Optional[CardinalityHashingModel] = None

    @model_validator(mode="after")
    def validate_options(self) -> "CardinalityLimitModel":
        if self.min_freq is not None and self.top_k is not None:
            raise ValueError(
                "top_k and min_freq are mutually exclusive cardinality limits"
            )
        if self.min_freq is None and self.top_k is None and self.hashing is None:
            raise ValueError(
                "cardinality configuration requires top_k, min_freq, or hashing"
            )
        return self


class PreprocessorModel(BaseModel):
    """Top-level preprocessing config."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    depth_layouts: DepthLayoutRegistryModel = Field(
        default_factory=DepthLayoutRegistryModel
    )
    project_root: str
    preprocessing_data_path: str
    read_format: str = "csv"
    write_format: str = "parquet"
    merge_output: bool = True
    allow_sequence_splitting: bool = False
    selected_columns: Optional[list[str]] = None
    categorical_columns: Optional[list[str]] = None
    real_columns: Optional[list[str]] = None
    column_data_types: Optional[dict[str, str]] = None
    normalize_real_columns: bool = True
    normalize_on_all_data: bool = False
    split_context: SplitContextConfig = Field(default_factory=SplitContextConfig)

    split_ratios: Optional[list[float]] = None
    split_method: str = Field(default="within_sequence")
    split_column: Optional[str] = None
    split_values: Optional[list[Union[int, str, date, datetime]]] = None
    window_length: int = Field(gt=0)
    max_target_offset: int = Field(default=1, ge=0)
    window_strides: Optional[list[int]] = None
    max_rows: Optional[int] = None
    seed: int = 1010
    n_cores: Optional[int] = None
    batches_per_file: int = 1024
    process_by_file: bool = True
    continue_preprocessing: bool = False
    window_placement: str = "distribute"
    use_precomputed_maps: Optional[list[str]] = None
    cardinality_config: dict[str, CardinalityLimitModel] = Field(default_factory=dict)
    metadata_config_path: Optional[str] = None
    mask_column: Optional[str] = None
    curriculum_column: Optional[Union[str, list[str]]] = None

    @field_validator("curriculum_column")
    @classmethod
    def validate_curriculum_columns(cls, value):
        columns = [value] if isinstance(value, str) else value
        if columns is not None and (
            not columns
            or any(not column for column in columns)
            or any(
                column.startswith("__sequifier_curriculum_value_") for column in columns
            )
            or len(columns) != len(set(columns))
        ):
            raise ValueError("curriculum_column must contain unique, non-empty names")
        return value

    @field_validator("preprocessing_data_path")
    @classmethod
    @beartype
    def validate_preprocessing_data_path(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError(f"{v} does not exist")
        return v

    @field_validator("read_format")
    @classmethod
    @beartype
    def validate_read_format(cls, v: str) -> str:
        supported_formats = ["csv", "parquet"]
        if v not in supported_formats:
            raise ValueError(
                f"Currently only {', '.join(supported_formats)} are supported "
                "for preprocessing input"
            )
        return v

    @field_validator("write_format")
    @classmethod
    @beartype
    def validate_write_format(cls, v: str) -> str:
        supported_formats = ["csv", "parquet", "pt"]
        if v not in supported_formats:
            raise ValueError(
                f"Currently only {', '.join(supported_formats)} are supported "
                "for preprocessing output"
            )
        return v

    @field_validator("merge_output")
    @classmethod
    @beartype
    def validate_format2(cls, v: bool, info: Any):
        write_format = info.data.get("write_format")

        if write_format == "pt" and v is True:
            raise ValueError(
                "With write_format 'pt', merge_output must be set to False"
            )

        if write_format == "parquet" and v is True:
            warnings.warn(
                "Training on distributed data in parquet format takes significantly more CPU per GPU than with 'pt'. Inferring on distributed data in parquet is less efficient than with 'pt'"
            )

        # Allow "parquet" to have merge_output = False
        if write_format not in ["pt", "parquet"] and v is False:
            raise ValueError(
                f"With write_format '{write_format}', merge_output must be set to True. "
                "Only 'pt' and 'parquet' formats support uncombined (split) output."
            )

        return v

    @field_validator("split_ratios")
    @classmethod
    @beartype
    def validate_proportions_sum(
        cls, v: Optional[list[float]]
    ) -> Optional[list[float]]:
        if v is None:
            return None
        if not np.isclose(np.sum(v), 1.0):
            raise ValueError(f"split_ratios must sum to 1.0, but sums to {np.sum(v)}")
        if not all(p > 0 for p in v):
            raise ValueError(f"All split_ratios must be positive: {v}")
        return v

    @field_validator("split_method")
    @classmethod
    @beartype
    def validate_split_method(cls, v: str) -> str:
        if v not in ["within_sequence", "between_sequence", "value_cutoff"]:
            raise ValueError(
                "split_method must be one of 'within_sequence', "
                "'between_sequence', 'value_cutoff'"
            )
        return v

    @field_validator("window_strides")
    @classmethod
    @beartype
    def validate_step_sizes(cls, v: Optional[list[int]], info: Any) -> list[int]:
        split_ratios = info.data.get("split_ratios")
        split_values = info.data.get("split_values")
        n_splits = (
            len(split_ratios)
            if split_ratios is not None
            else len(split_values) + 1
            if split_values is not None
            else None
        )
        if n_splits is None:
            raise ValueError(
                "split_ratios or split_values must be set to validate window_strides"
            )

        if not isinstance(v, list):
            raise ValueError("window_strides should be a list after __init__")

        if len(v) != n_splits:
            raise ValueError(
                f"Length of window_strides ({len(v)}) must match length of "
                f"configured splits ({n_splits})"
            )
        if not all(step > 0 for step in v):
            raise ValueError(f"All window_strides must be positive integers: {v}")
        return v

    @field_validator("batches_per_file")
    @classmethod
    @beartype
    def validate_batches_per_file(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batches_per_file must be a positive integer")
        return v

    @field_validator("column_data_types")
    @classmethod
    @beartype
    def validate_column_types(
        cls, v: Optional[dict[str, str]], info: Any
    ) -> Optional[dict[str, str]]:
        if not v:
            return None

        normalized = {
            column: (
                dtype
                if dtype.startswith(("Date", "Datetime"))
                else canonicalize_polars_dtype_name(dtype)
            )
            for column, dtype in v.items()
        }
        selected_columns = info.data.get("selected_columns")
        if selected_columns is not None:
            missing_columns = [
                column for column in selected_columns if column not in normalized
            ]
            if missing_columns:
                raise ValueError(
                    "column_data_types must include every selected column. "
                    f"Missing: {missing_columns}"
                )

        return normalized

    @field_validator("categorical_columns", "real_columns")
    @classmethod
    @beartype
    def validate_column_roles(
        cls, value: Optional[list[str]], info: Any
    ) -> Optional[list[str]]:
        if value is None:
            return None
        if any(not column for column in value) or len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} must contain unique, non-empty names")
        return value

    @field_validator("continue_preprocessing")
    @classmethod
    @beartype
    def validate_continue_preprocessing(cls, v: bool, info: Any) -> bool:
        if v and info.data.get("merge_output"):
            raise ValueError(
                "'continue_preprocessing' can only be set to true if "
                "merge_output is False, not single files"
            )
        return v

    @field_validator("window_placement")
    @classmethod
    @beartype
    def validate_window_placement(cls, v: str) -> str:
        if v not in ["distribute", "exact"]:
            raise ValueError("window_placement must be one of 'distribute', 'exact'")
        return v

    @model_validator(mode="after")
    @beartype
    def validate_mask_column_requires_metadata(self) -> "PreprocessorModel":
        categorical = set(self.categorical_columns or [])
        real = set(self.real_columns or [])
        overlap = categorical & real
        if overlap:
            raise ValueError(
                "Columns cannot be both categorical and real: " f"{sorted(overlap)}"
            )

        declared = categorical | real
        cardinality_columns = set(self.cardinality_config)
        if cardinality_columns & real:
            raise ValueError(
                "cardinality_config may only reference categorical columns. "
                f"Real columns: {sorted(cardinality_columns & real)}"
            )
        if categorical and cardinality_columns - categorical:
            raise ValueError(
                "When categorical_columns is set, cardinality_config columns must "
                f"be categorical. Invalid: {sorted(cardinality_columns - categorical)}"
            )
        if self.selected_columns is not None:
            unselected_cardinality = cardinality_columns - set(self.selected_columns)
            if unselected_cardinality:
                raise ValueError(
                    "cardinality_config columns must be selected columns. "
                    f"Not selected: {sorted(unselected_cardinality)}"
                )
        precomputed_overlap = cardinality_columns & set(self.use_precomputed_maps or [])
        if precomputed_overlap:
            raise ValueError(
                "cardinality_config cannot be combined with precomputed maps "
                "(use_precomputed_maps) for the same columns: "
                f"{sorted(precomputed_overlap)}"
            )
        if self.selected_columns is not None:
            unknown = declared - set(self.selected_columns)
            if unknown:
                raise ValueError(
                    "categorical_columns and real_columns must be selected columns. "
                    f"Not selected: {sorted(unknown)}"
                )

        if self.column_data_types is not None:
            missing_types = declared - set(self.column_data_types)
            if missing_types:
                raise ValueError(
                    "column_data_types must include every explicitly classified "
                    f"column. Missing: {sorted(missing_types)}"
                )
            categorical_with_non_integer_types = sorted(
                column
                for column in categorical
                if not is_integer_dtype_name(self.column_data_types[column])
            )
            if categorical_with_non_integer_types:
                raise ValueError(
                    "Categorical columns require integer column_data_types. "
                    f"Invalid: {categorical_with_non_integer_types}"
                )
            real_with_non_float_types = sorted(
                column
                for column in real
                if not is_float_dtype_name(self.column_data_types[column])
            )
            if real_with_non_float_types:
                raise ValueError(
                    "Real columns require floating-point column_data_types. "
                    f"Invalid: {real_with_non_float_types}"
                )

        if (self.split_values is None) == (self.split_ratios is None):
            raise ValueError("Exactly one of split_values and split_ratios must be set")
        if self.split_method == "value_cutoff":
            if not self.split_column:
                raise ValueError(
                    "split_column must be set when split_method is 'value_cutoff'"
                )
            if not self.split_values:
                raise ValueError(
                    "split_values must be set when split_method is 'value_cutoff'"
                )
            if self.split_ratios is not None:
                raise ValueError(
                    "split_ratios must be null when split_method is 'value_cutoff'"
                )
            if self.split_column in {"sequenceId", "itemPosition"}:
                raise ValueError("split_column cannot be sequenceId or itemPosition")
            if self.split_column == "__sequifier_split_value":
                raise ValueError(
                    "split_column cannot use reserved name " "'__sequifier_split_value'"
                )
            if self.split_column == self.mask_column:
                raise ValueError("split_column cannot also be mask_column")
            curriculum_columns = (
                [self.curriculum_column]
                if isinstance(self.curriculum_column, str)
                else self.curriculum_column or []
            )
            if self.split_column in curriculum_columns:
                raise ValueError("split_column cannot also be a curriculum_column")
            temporal_columns = {
                column
                for column, dtype in (self.column_data_types or {}).items()
                if dtype.startswith(("Date", "Datetime"))
            }
            if temporal_columns - {self.split_column}:
                raise ValueError(
                    "Only split_column may have a timestamp column_data_type"
                )
        elif self.split_values is not None:
            raise ValueError(
                "split_values may only be set when split_method is 'value_cutoff'"
            )
        elif self.split_ratios is None:
            raise ValueError("split_ratios must be set for ratio-based split methods")
        elif self.split_column is not None:
            raise ValueError(
                "split_column is only valid when split_method is 'value_cutoff'"
            )
        elif any(
            dtype.startswith(("Date", "Datetime"))
            for dtype in (self.column_data_types or {}).values()
        ):
            raise ValueError(
                "Timestamp columns are only supported as value_cutoff split_column"
            )
        if self.depth_layouts:
            if self.split_method == "value_cutoff":
                raise ValueError(
                    "value_cutoff splitting is not supported with depth_layouts"
                )
            if (
                self.write_format != "pt"
                or self.merge_output
                or self.mask_column is not None
            ):
                raise ValueError(
                    "Depth preprocessing requires write_format: pt, merge_output: false, and no mask_column"
                )
            if len(self.depth_layouts.root) != 1:
                raise ValueError("Raw preprocessing currently accepts one depth layout")
            if self.selected_columns is not None and not set(
                self.depth_layouts.deep_columns
            ) <= set(self.selected_columns):
                raise ValueError("selected_columns must include every depth feature")
            positions = {
                layout.position_column for _, layout in self.depth_layouts.items()
            }
            if positions.intersection(
                self.selected_columns or []
            ) or positions.intersection(self.column_data_types or {}):
                raise ValueError(
                    "Depth position columns are coordinates, not selected features or column_data_types"
                )
        if self.mask_column is not None and self.metadata_config_path is None:
            raise ValueError("metadata_config_path must be set when mask_column is set")
        if self.mask_column in ("sequenceId", "itemPosition"):
            raise ValueError("mask_column cannot be sequenceId or itemPosition")
        curriculum_columns = (
            [self.curriculum_column]
            if isinstance(self.curriculum_column, str)
            else self.curriculum_column or []
        )
        if set(curriculum_columns) & {"sequenceId", "itemPosition"}:
            raise ValueError("curriculum_column cannot be sequenceId or itemPosition")
        if curriculum_columns and (
            self.mask_column in curriculum_columns
            or set(curriculum_columns)
            & {layout.position_column for _, layout in self.depth_layouts.items()}
            or set(curriculum_columns) & set(self.depth_layouts.deep_columns)
        ):
            raise ValueError(
                "curriculum_column cannot also be a mask or depth feature/position "
                "column"
            )
        if self.max_target_offset >= self.window_length:
            raise ValueError("max_target_offset must be smaller than window_length")
        if self.split_context.mode == "preceding":
            if self.split_method not in {"within_sequence", "value_cutoff"}:
                raise ValueError(
                    "split_context preceding mode requires split_method: "
                    "within_sequence or value_cutoff"
                )
            assert self.split_context.target_offset is not None
            if self.split_context.target_offset > self.max_target_offset:
                raise ValueError(
                    "split_context target_offset cannot exceed max_target_offset"
                )
            self.split_context.halo_length(self.window_length, self.max_target_offset)
            if self.allow_sequence_splitting:
                raise ValueError(
                    "split_context preceding mode requires "
                    "allow_sequence_splitting: false so history is preserved"
                )
        return self

    @beartype
    def __init__(self, **kwargs):
        split_ratios = kwargs.get("split_ratios")
        split_values = kwargs.get("split_values")
        n_splits = (
            len(split_ratios)
            if split_ratios is not None
            else len(split_values) + 1
            if split_values is not None
            else 0
        )
        if n_splits:
            default_stride_for_split = [kwargs["window_length"]] * n_splits
            kwargs["window_strides"] = kwargs.get(
                "window_strides", default_stride_for_split
            )
        super().__init__(**kwargs)
