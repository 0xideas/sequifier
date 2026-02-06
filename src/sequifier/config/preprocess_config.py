import os
from typing import Optional

import numpy as np
import yaml
from beartype import beartype
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

from sequifier.helpers import try_catch_excess_keys


@beartype
def load_preprocessor_config(
    config_path: str, args_config: dict
) -> "PreprocessorModel":
    """
    Load preprocessor configuration from a YAML file and update it with args_config.

    Args:
        config_path: Path to the YAML configuration file.
        args_config: Dictionary containing additional configuration arguments.

    Returns:
        PreprocessorModel instance with loaded configuration.
    """
    with open(config_path, "r") as f:
        config_values = yaml.safe_load(f)

    config_values.update(args_config)

    config_values["seed"] = config_values.get("seed", 1010)

    return try_catch_excess_keys(config_path, PreprocessorModel, config_values)


class PreprocessorModel(BaseModel):
    """
    Pydantic model for preprocessor configuration.

    Attributes:
        project_root: The path to the sequifier project directory.
        data_path: The path to the input data file.
        read_format: The file type of the input data. Can be 'csv' or 'parquet'.
        write_format: The file type for the preprocessed output data.
        merge_output: If True, combines all preprocessed data into a single file.
        selected_columns: A list of columns to be included in the preprocessing. If None, all columns are used.
        split_ratios: A list of floats that define the relative sizes of data splits (e.g., for train, validation, test).
                           The sum of proportions must be 1.0.
        seq_length: The sequence length for the model inputs.
        stride_by_split: A list of step sizes for creating subsequences within each data split.
        max_rows: The maximum number of input rows to process. If None, all rows are processed.
        seed: A random seed for reproducibility.
        n_cores: The number of CPU cores to use for parallel processing. If None, it uses the available CPU cores.
        batches_per_file: The number of batches to process per file.
        process_by_file: A flag to indicate if processing should be done file by file.
        continue_preprocessing: Continue preprocessing job that was interrupted while writing to temp folder.
        subsequence_start_mode: "distribute" to minimize max subsequence overlap, or "exact".
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    project_root: str
    data_path: str
    read_format: str = "csv"
    write_format: str = "parquet"
    merge_output: bool = True
    selected_columns: Optional[list[str]] = None

    split_ratios: list[float]
    seq_length: int
    stride_by_split: Optional[list[int]] = None
    max_rows: Optional[int] = None
    seed: int
    n_cores: Optional[int] = None
    batches_per_file: int = 1024
    process_by_file: bool = True
    continue_preprocessing: bool = False
    subsequence_start_mode: str = "distribute"
    use_precomputed_maps: Optional[list[str]] = None

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

        # Existing check: 'pt' format cannot be combined
        if write_format == "pt" and v is True:
            raise ValueError(
                "With write_format 'pt', merge_output must be set to False"
            )

        # New constraint: 'parquet' or 'csv' formats cannot be uncombined (split)
        if write_format != "pt" and v is False:
            raise ValueError(
                f"With write_format '{write_format}', merge_output must be set to True. "
                "Only 'pt' format supports uncombined (split) output."
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

    @field_validator("continue_preprocessing")
    @classmethod
    def validate_continue_preprocessing(cls, v: bool, info: ValidationInfo) -> bool:
        if v and info.data.get("data_path").split(".") in ["csv", "parquet"]:
            raise ValueError(
                "'continue_preprocessing' can only be set to true for folder inputs, not single files "
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

    def __init__(self, **kwargs):
        default_stride_for_split = [kwargs["seq_length"]] * len(kwargs["split_ratios"])
        kwargs["stride_by_split"] = kwargs.get(
            "stride_by_split", default_stride_for_split
        )
        super().__init__(**kwargs)
