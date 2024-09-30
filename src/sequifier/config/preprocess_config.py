import os
from typing import Optional

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
    seq_step_size: Optional[int]
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

    def __init__(self, **kwargs):
        kwargs["seq_step_size"] = kwargs.get("seq_step_size", kwargs["seq_length"])
        kwargs["seq_length"] += 1
        super().__init__(**kwargs)
