import os
import warnings
from typing import Optional

import numpy as np
import yaml
from beartype import beartype
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from sequifier.helpers import canonicalize_polars_dtype_name, try_catch_excess_keys


@beartype
def load_preprocessor_config(
    config_path: str, args_config: dict
) -> "PreprocessorModel":
    """Load preprocessing YAML plus CLI overrides."""
    with open(config_path, "r") as f:
        config_values = yaml.safe_load(f)

    config_values.update(args_config)

    config_values["seed"] = config_values.get("seed", 1010)

    return try_catch_excess_keys(config_path, PreprocessorModel, config_values)


class PreprocessorModel(BaseModel):
    """Top-level preprocessing config."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    project_root: str
    data_path: str
    read_format: str = "csv"
    write_format: str = "parquet"
    merge_output: bool = True
    allow_sequence_splitting: bool = False
    selected_columns: Optional[list[str]] = None
    column_types: Optional[dict[str, str]] = None

    split_ratios: list[float]
    stored_context_width: int = Field(gt=0)
    max_target_offset: int = Field(default=1, ge=0)
    stride_by_split: Optional[list[int]] = None
    max_rows: Optional[int] = None
    seed: int
    n_cores: Optional[int] = None
    batches_per_file: int = 1024
    process_by_file: bool = True
    continue_preprocessing: bool = False
    subsequence_start_mode: str = "distribute"
    use_precomputed_maps: Optional[list[str]] = None
    metadata_config_path: Optional[str] = None
    mask_column: Optional[str] = None

    @field_validator("data_path")
    @classmethod
    def validate_data_path(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError(f"{v} does not exist")
        return v

    @field_validator("read_format", "write_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        supported_formats = ["csv", "parquet", "pt"]
        if v not in supported_formats:
            raise ValueError(
                f"Currently only {', '.join(supported_formats)} are supported"
            )
        return v

    @field_validator("merge_output")
    @classmethod
    def validate_format2(cls, v: bool, info: ValidationInfo):
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
    def validate_proportions_sum(cls, v: list[float]) -> list[float]:
        if not np.isclose(np.sum(v), 1.0):
            raise ValueError(f"split_ratios must sum to 1.0, but sums to {np.sum(v)}")
        if not all(p > 0 for p in v):
            raise ValueError(f"All split_ratios must be positive: {v}")
        return v

    @field_validator("stride_by_split")
    @classmethod
    def validate_step_sizes(
        cls, v: Optional[list[int]], info: ValidationInfo
    ) -> list[int]:
        split_ratios = info.data.get("split_ratios")
        if not (split_ratios is not None):
            raise ValueError("split_ratios must be set to validate stride_by_split")

        if not isinstance(v, list):
            raise ValueError("stride_by_split should be a list after __init__")

        if len(v) != len(split_ratios):
            raise ValueError(
                f"Length of stride_by_split ({len(v)}) must match length of "
                f"split_ratios ({len(split_ratios)})"
            )
        if not all(step > 0 for step in v):
            raise ValueError(f"All stride_by_split must be positive integers: {v}")
        return v

    @field_validator("batches_per_file")
    @classmethod
    def validate_batches_per_file(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batches_per_file must be a positive integer")
        return v

    @field_validator("column_types")
    @classmethod
    def validate_column_types(
        cls, v: Optional[dict[str, str]], info: ValidationInfo
    ) -> Optional[dict[str, str]]:
        if v is None:
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
                    "column_types must include every selected column. "
                    f"Missing: {missing_columns}"
                )

        return normalized

    @field_validator("continue_preprocessing")
    @classmethod
    def validate_continue_preprocessing(cls, v: bool, info: ValidationInfo) -> bool:
        if v and info.data.get("merge_data"):
            raise ValueError(
                "'continue_preprocessing' can only be set to true if merge_data is False, not single files "
            )
        return v

    @field_validator("subsequence_start_mode")
    @classmethod
    def validate_subsequence_start_mode(cls, v: str) -> str:
        if v not in ["distribute", "exact"]:
            raise ValueError(
                "subsequence_start_mode must be one of 'distribute', 'exact'"
            )
        return v

    @model_validator(mode="after")
    def validate_mask_column_requires_metadata(self) -> "PreprocessorModel":
        if self.mask_column is not None and self.metadata_config_path is None:
            raise ValueError("metadata_config_path must be set when mask_column is set")
        if self.mask_column in ("sequenceId", "itemPosition"):
            raise ValueError("mask_column cannot be sequenceId or itemPosition")
        if self.max_target_offset >= self.stored_context_width:
            raise ValueError(
                "max_target_offset must be smaller than stored_context_width"
            )
        return self

    def __init__(self, **kwargs):
        default_stride_for_split = [kwargs["stored_context_width"]] * len(
            kwargs["split_ratios"]
        )
        kwargs["stride_by_split"] = kwargs.get(
            "stride_by_split", default_stride_for_split
        )
        super().__init__(**kwargs)
