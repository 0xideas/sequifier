import os
import warnings
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sequifier.config.composition import (
    load_composed_yaml_config,
    merge_config_fragments,
)
from sequifier.helpers import canonicalize_polars_dtype_name, try_catch_excess_keys
from sequifier.typechecking import beartype


@beartype
def load_preprocessor_config(
    config_path: str, args_config: dict
) -> "PreprocessorModel":
    """Load preprocessing YAML plus CLI overrides."""
    config_values = load_composed_yaml_config(config_path)

    config_values = merge_config_fragments((config_values, args_config))

    return try_catch_excess_keys(config_path, PreprocessorModel, config_values)


class PreprocessorModel(BaseModel):
    """Top-level preprocessing config."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    project_root: str
    preprocessing_data_path: str
    read_format: str = "csv"
    write_format: str = "parquet"
    merge_output: bool = True
    allow_sequence_splitting: bool = False
    selected_columns: Optional[list[str]] = None
    column_data_types: Optional[dict[str, str]] = None
    normalize_real_columns: bool = True

    split_ratios: list[float]
    split_method: str = Field(default="within_sequence")
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
    metadata_config_path: Optional[str] = None
    mask_column: Optional[str] = None

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
    def validate_proportions_sum(cls, v: list[float]) -> list[float]:
        if not np.isclose(np.sum(v), 1.0):
            raise ValueError(f"split_ratios must sum to 1.0, but sums to {np.sum(v)}")
        if not all(p > 0 for p in v):
            raise ValueError(f"All split_ratios must be positive: {v}")
        return v

    @field_validator("split_method")
    @classmethod
    @beartype
    def validate_split_method(cls, v: str) -> str:
        if v not in ["within_sequence", "between_sequence"]:
            raise ValueError(
                "split_method must be one of 'within_sequence', 'between_sequence'"
            )
        return v

    @field_validator("window_strides")
    @classmethod
    @beartype
    def validate_step_sizes(cls, v: Optional[list[int]], info: Any) -> list[int]:
        split_ratios = info.data.get("split_ratios")
        if not (split_ratios is not None):
            raise ValueError("split_ratios must be set to validate window_strides")

        if not isinstance(v, list):
            raise ValueError("window_strides should be a list after __init__")

        if len(v) != len(split_ratios):
            raise ValueError(
                f"Length of window_strides ({len(v)}) must match length of "
                f"split_ratios ({len(split_ratios)})"
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
            column: canonicalize_polars_dtype_name(dtype) for column, dtype in v.items()
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
        if self.mask_column is not None and self.metadata_config_path is None:
            raise ValueError("metadata_config_path must be set when mask_column is set")
        if self.mask_column in ("sequenceId", "itemPosition"):
            raise ValueError("mask_column cannot be sequenceId or itemPosition")
        if self.max_target_offset >= self.window_length:
            raise ValueError("max_target_offset must be smaller than window_length")
        return self

    @beartype
    def __init__(self, **kwargs):
        default_stride_for_split = [kwargs["window_length"]] * len(
            kwargs["split_ratios"]
        )
        kwargs["window_strides"] = kwargs.get(
            "window_strides", default_stride_for_split
        )
        super().__init__(**kwargs)
