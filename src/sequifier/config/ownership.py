"""Run resources and durable model contracts, independent of learning algorithms."""

from typing import Annotated, ClassVar, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from sequifier.config.cardinality import CardinalityLimitModel
from sequifier.config.components import ResumeConfig
from sequifier.config.naming import ConfigModel, RunSeed, SeedInteger
from sequifier.config.normalization_config import FrozenNormalizationConfig
from sequifier.model.execution_options import DEFAULT_MAX_CONCAT_TENSORS, ConcatLimit


class Settings(ConfigModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class LocalDistribution(Settings):
    strategy: Literal["local"] = "local"
    distributed: ClassVar[bool] = False
    world_size: ClassVar[int] = 1


class DistributedSettings(Settings):
    distributed: ClassVar[bool] = True
    world_size: int = Field(default=1, gt=0)
    backend: str = "nccl"


class DDPDistribution(DistributedSettings):
    strategy: Literal["ddp"]


class FSDPDistribution(DistributedSettings):
    strategy: Literal["fsdp"]
    cpu_offload: bool


DistributionSettings = Annotated[
    LocalDistribution | DDPDistribution | FSDPDistribution,
    Field(discriminator="strategy"),
]


class PrecisionSettings(Settings):
    layer_type_dtypes: dict[str, str] | None = None
    layer_autocast: bool = False
    float32_matmul_precision: Literal["highest", "high", "medium"] = "highest"


CompileMode = Literal["outer", "inner", "none"]


class CompilationSettings(Settings):
    torch_compile: CompileMode = "outer"


class PolicyCompilationSettings(Settings):
    torch_compile: CompileMode = "none"


class PolicyExecution(Settings):
    compilation: PolicyCompilationSettings = Field(
        default_factory=PolicyCompilationSettings
    )


class DataLoadingSettings(Settings):
    num_workers: int = Field(default=0, ge=0)
    load_full_data_to_ram: bool = True
    max_ram_gb: float = Field(default=16, gt=0)


class SupervisedExecution(Settings):
    distribution: DistributionSettings = Field(default_factory=LocalDistribution)
    precision: PrecisionSettings = Field(default_factory=PrecisionSettings)
    compilation: CompilationSettings = Field(default_factory=CompilationSettings)
    data_loading: DataLoadingSettings = Field(default_factory=DataLoadingSettings)
    max_concat_tensors: ConcatLimit = DEFAULT_MAX_CONCAT_TENSORS

    @model_validator(mode="after")
    def validate_compilation(self):
        strategy = self.distribution.strategy
        mode = self.compilation.torch_compile
        if strategy == "fsdp" and mode == "outer":
            raise ValueError("FSDP requires torch_compile 'none' or 'inner'")
        if strategy == "ddp" and mode == "inner":
            raise ValueError("DDP requires torch_compile 'none' or 'outer'")
        return self


class OfflineExecution(Settings):
    episode_cache_size: int = Field(default=8, gt=0)


class ExecutionSettings(Settings):
    device: str = "cpu"
    supervised: SupervisedExecution | None = None
    offline: OfflineExecution | None = None
    policy: PolicyExecution | None = None
    distribution: DistributionSettings | None = None

    @model_validator(mode="after")
    def resolve_distribution(self):
        if self.distribution is not None and self.supervised is not None:
            supervised = self.supervised
            if (
                "distribution" in supervised.model_fields_set
                and supervised.distribution != self.distribution
            ):
                raise ValueError("Conflicting run and supervised distribution settings")
            self.supervised = SupervisedExecution.model_validate(
                {
                    **supervised.model_dump(exclude_unset=True),
                    "distribution": self.distribution,
                }
            )
        return self

    @property
    def distribution_settings(self):
        if self.distribution is not None:
            return self.distribution
        return self.supervised.distribution if self.supervised else LocalDistribution()

    @property
    def policy_compile_mode(self) -> CompileMode:
        return self.policy.compilation.torch_compile if self.policy else "none"

    @property
    def supervised_settings(self) -> SupervisedExecution:
        if self.supervised is None:
            raise ValueError("This operation requires supervised execution")
        return self.supervised


class _DeterminismSettings(Settings):
    enforce_determinism: bool = False


class TrainingReproducibilitySettings(_DeterminismSettings):
    seed: SeedInteger | Literal["random"] = 1010


class ReproducibilitySettings(_DeterminismSettings):
    seed: RunSeed = 1010


class RecoverySettings(Settings):
    resume: ResumeConfig = Field(default_factory=ResumeConfig)
    latest_checkpoint_interval_minutes: float | None = Field(default=None, gt=0)


class LoggingSettings(Settings):
    log_interval_batches: int | None = Field(default=None, gt=0)
    log_interval_iterations: int | None = Field(default=None, gt=0)


class OutputSettings(Settings):
    model_path: str = ""
    report_path: str = ""
    phase_directory: str = ""


class PortableExport(Settings):
    """Portable artifacts carry their execution contract."""


class GraphExport(Settings):
    with_dropout: bool = False
    batch_size: int = Field(default=1, gt=0)


class ArtifactFormats(Settings):
    pt: PortableExport | None = None
    onnx: GraphExport | None = None

    @model_validator(mode="after")
    def require_format(self):
        if self.pt is None and self.onnx is None:
            raise ValueError("Select an artifact format (pt or onnx)")
        return self


class EmbeddingExport(ArtifactFormats):
    layers: list[str] = Field(
        default_factory=lambda: ["backbone.final_norm"], min_length=1
    )


class ExportSettings(Settings):
    prediction: ArtifactFormats | None = None
    embedding: EmbeddingExport | None = None

    @property
    def embedding_layer_names(self) -> list[str]:
        return self.embedding.layers if self.embedding else ["backbone.final_norm"]


def portable_export():
    return ExportSettings(prediction=ArtifactFormats(pt=PortableExport()))


class RunSettings(Settings):
    retrain: bool = False
    project_root: str = "."
    model_name: str
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    reproducibility: TrainingReproducibilitySettings = Field(
        default_factory=TrainingReproducibilitySettings
    )
    recovery: RecoverySettings = Field(default_factory=RecoverySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    outputs: OutputSettings = Field(default_factory=OutputSettings)
    export: ExportSettings = Field(default_factory=portable_export)

    def effective_resume_policy(self):
        if self.retrain:
            return "never"
        return "required" if self.recovery.resume.policy == "required" else "if_exists"

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value):
        if "embedding" in value:
            raise ValueError("model_name cannot contain 'embedding'")
        return value


class ModelExecutionSettings(Settings):
    context_length: int | None = Field(default=None, gt=0)
    target_offset: int = Field(default=1, ge=0)
    prediction_view: (
        Literal["causal", "policy", "final_value", "bert", "next_occurrence"] | None
    ) = None
    forward_attention: Literal["causal", "bidirectional"] | None = None


class InputContract(Settings):
    metadata_config_path: str | None = None
    normalization: FrozenNormalizationConfig | None = None
    cardinality_config: dict[str, CardinalityLimitModel] | None = None

    @property
    def cardinality_rules(self):
        return {
            name: rule.model_dump(exclude_none=True)
            for name, rule in (self.cardinality_config or {}).items()
        }


class SplitSettings(Settings):
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
