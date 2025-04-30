import os
from typing import Optional

import numpy as np
import yaml
from beartype import beartype
from pydantic import BaseModel, validator


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

    return PreprocessorModel(**config_values)


class PreprocessorModel(BaseModel):
    """
    Pydantic model for preprocessor configuration.
    """

    project_path: str
    data_path: str
    read_format: str = "csv"
    write_format: str = "parquet"
    selected_columns: Optional[list[str]]

    group_proportions: list[float]
    seq_length: int
    seq_step_sizes: Optional[list[int]]
    max_rows: Optional[int]
    seed: int
    n_cores: Optional[int]

    @validator("data_path", always=True)
    def validate_data_path(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError(f"{v} does not exist")
        return v

    @validator("read_format", "write_format", always=True)
    def validate_format(cls, v: str) -> str:
        supported_formats = ["csv", "parquet"]
        if v not in supported_formats:
            raise ValueError(
                f"Currently only {', '.join(supported_formats)} are supported"
            )
        return v

    @validator("group_proportions")
    def validate_proportions_sum(cls, v: list[float]) -> list[float]:
        if not np.isclose(np.sum(v), 1.0):
            raise ValueError(
                f"group_proportions must sum to 1.0, but sums to {np.sum(v)}"
            )
        if not all(p > 0 for p in v):
            raise ValueError(f"All group_proportions must be positive: {v}")
        return v

    @validator("seq_step_sizes", always=True)
    def validate_step_sizes(cls, v: Optional[list[int]], values: dict) -> list[int]:
        group_proportions = values.get("group_proportions")
        assert (
            group_proportions is not None
        ), "group_proportions must be set to validate seq_step_sizes"

        assert isinstance(v, list), "seq_step_sizes should be a list after __init__"

        if len(v) != len(group_proportions):
            raise ValueError(
                f"Length of seq_step_sizes ({len(v)}) must match length of "
                f"group_proportions ({len(group_proportions)})"
            )
        if not all(step > 0 for step in v):
            raise ValueError(f"All seq_step_sizes must be positive integers: {v}")
        return v

    def __init__(self, **kwargs):
        default_seq_step_size = [kwargs["seq_length"]] * len(
            kwargs["group_proportions"]
        )
        kwargs["seq_step_sizes"] = kwargs.get("seq_step_sizes", default_seq_step_size)
        kwargs["seq_length"] += 1
        super().__init__(**kwargs)
