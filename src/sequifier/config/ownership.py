"""Run resources and durable model contracts, independent of learning algorithms."""

from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from sequifier.config.components import NextOccurrenceConfigModel, ResumeConfig
from sequifier.config.naming import ConfigModel, RunSeed
from sequifier.config.normalization_config import FrozenNormalizationConfig


class Settings(ConfigModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DistributionSettings(Settings):
    strategy: Literal["local", "ddp", "fsdp"] = "local"
    world_size: int = Field(default=1, gt=0)
    backend: str = "nccl"
    cpu_offload: bool | None = None

    @model_validator(mode="after")
    def coherent(self):
        if self.strategy == "local" and self.world_size != 1:
            raise ValueError("local distribution requires world_size: 1")
        if self.strategy != "fsdp" and self.cpu_offload is not None:
            raise ValueError("cpu_offload is only valid with fsdp")
        if self.strategy == "fsdp" and self.cpu_offload is None:
            raise ValueError("fsdp requires explicit cpu_offload")
        return self


class PrecisionSettings(Settings):
    layer_type_dtypes: dict[str, str] | None = None
    layer_autocast: bool = False
    float32_matmul_precision: Literal["highest", "high", "medium"] = "highest"


class CompilationSettings(Settings):
    torch_compile: Literal["outer", "inner", "none"] = "none"


class DataLoadingSettings(Settings):
    num_workers: int = Field(default=0, ge=0)
    load_full_data_to_ram: bool = True
    max_ram_gb: float = Field(default=16, gt=0)
    episode_cache_size: int = Field(default=8, gt=0)


class ExecutionSettings(Settings):
    device: str = "cpu"
    distribution: DistributionSettings = Field(default_factory=DistributionSettings)
    precision: PrecisionSettings = Field(default_factory=PrecisionSettings)
    compilation: CompilationSettings = Field(default_factory=CompilationSettings)
    data_loading: DataLoadingSettings = Field(default_factory=DataLoadingSettings)
    max_concat_tensors: int = Field(default=12, gt=0)


class ReproducibilitySettings(Settings):
    seed: RunSeed = 1010
    enforce_determinism: bool = False


class RecoverySettings(Settings):
    resume: ResumeConfig = Field(default_factory=ResumeConfig)
    latest_checkpoint_interval_minutes: float | None = Field(default=None, gt=0)


class LoggingSettings(Settings):
    log_interval_batches: int = Field(default=10, gt=0)


class OutputSettings(Settings):
    model_path: str = ""
    report_path: str = ""
    phase_directory: str = ""


class ExportSettings(Settings):
    generative_model: bool = True
    embedding_model: bool = False
    embedding_layer_names: list[str] = Field(
        default_factory=lambda: ["backbone.final_norm"], min_length=1
    )
    onnx: bool = False
    pt: bool = True
    with_dropout: bool = False
    batch_size: int = Field(default=1, gt=0)


class RunSettings(Settings):
    project_root: str = "."
    model_name: str
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    reproducibility: ReproducibilitySettings = Field(
        default_factory=ReproducibilitySettings
    )
    recovery: RecoverySettings = Field(default_factory=RecoverySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    outputs: OutputSettings = Field(default_factory=OutputSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value):
        if "embedding" in value:
            raise ValueError("model_name cannot contain 'embedding'")
        return value


class ModelExecutionSettings(Settings):
    context_length: int | None = Field(default=None, gt=0)
    target_offset: int = Field(default=1, ge=0)
    training_objective: str | None = None
    forward_attention: Literal["causal", "bidirectional"] | None = None
    next_occurrence_config: NextOccurrenceConfigModel | None = None


class InputContract(Settings):
    metadata_config_path: str | None = None
    normalization: FrozenNormalizationConfig | None = None


class SplitSettings(Settings):
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
