"""Execution options consumed by every inference route."""

from pydantic import BaseModel, ConfigDict, Field


class InferenceExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    project_root: str = "."
    model_interface: str | None = None
    read_format: str = "parquet"
    write_format: str = "csv"
    device: str = "cpu"
    seed: int = 1010
    enforce_determinism: bool = False
    inference_batch_size: int = Field(default=1, gt=0)
