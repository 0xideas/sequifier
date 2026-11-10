"""A common authored run plan with supervised and policy phases."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sequifier.config.policy_config import PolicySchemaConfig
from sequifier.config.policy_plan import (
    DPOPhaseConfig,
    GRPOPhaseConfig,
    IQLPhaseConfig,
    PolicyImitationPhaseConfig,
    PolicyResumeConfig,
    PPOPhaseConfig,
)
from sequifier.config.policy_train_config import SequenceSourceConfig
from sequifier.config.train_config import (
    DatasetTrainingSpecModel,
    EvaluationSpecModel,
    GlobalTrainingSpecModel,
    ModelSpecModel,
    TrainingPhaseSpecModel,
)


class SupervisedLearningPhase(TrainingPhaseSpecModel):
    handoff: Literal["best", "last"] = "best"


LearningPhase = Annotated[
    SupervisedLearningPhase
    | PolicyImitationPhaseConfig
    | IQLPhaseConfig
    | PPOPhaseConfig
    | DPOPhaseConfig
    | GRPOPhaseConfig,
    Field(discriminator="algorithm"),
]


class LearningRunConfig(BaseModel):
    """One model with typed sources and independently budgeted phase runners."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    run_type: Literal["unified"] = "unified"
    project_root: str = "."
    model_name: str = Field(min_length=1, pattern=r"^[^/\\]+$")
    device: str = "cpu"
    seed: int = Field(default=1010, ge=0, lt=2**32)
    model: ModelSpecModel | None = None
    initialize_from: str | None = None
    global_training: GlobalTrainingSpecModel | None = None
    dataset_training: dict[str, DatasetTrainingSpecModel] = Field(default_factory=dict)
    evaluation: EvaluationSpecModel | None = None
    policy_interface: str | None = None
    policy_bindings: dict[str, PolicySchemaConfig] = Field(default_factory=dict)
    sources: dict[str, SequenceSourceConfig] = Field(default_factory=dict)
    preferences: dict[str, str] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    phases: list[LearningPhase] = Field(min_length=1)
    resume: PolicyResumeConfig = Field(default_factory=PolicyResumeConfig)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    episode_cache_size: int = Field(default=8, gt=0)
    checkpoint_path: str = ""
    model_path: str = ""
    report_path: str = ""
    phase_directory: str = ""
    export_onnx: bool = False

    @model_validator(mode="after")
    def validate_plan(self):
        root = Path(self.project_root).resolve()
        self.project_root = str(root)
        if self.model_name in {".", ".."}:
            raise ValueError("model_name must name one directory")
        for key, default in (
            ("checkpoint_path", f"checkpoints/{self.model_name}/learning-run.pt"),
            ("model_path", f"models/{self.model_name}/model.pt"),
            ("report_path", f"outputs/training/{self.model_name}/report.json"),
            ("phase_directory", f"outputs/training/{self.model_name}/phases"),
        ):
            setattr(self, key, str((root / (getattr(self, key) or default)).resolve()))
        if self.initialize_from:
            self.initialize_from = str((root / self.initialize_from).resolve())
        if self.model is None and self.initialize_from is None:
            raise ValueError("a run requires a model or initialize_from")
        if len({phase.name for phase in self.phases}) != len(self.phases):
            raise ValueError("phase names must be unique")
        if any(phase.algorithm == "supervised" for phase in self.phases):
            if self.global_training is None or not self.dataset_training:
                raise ValueError(
                    "supervised phases require global_training and dataset_training"
                )
        if self.phases[0].algorithm != "supervised" and self.initialize_from is None:
            raise ValueError("policy-first unified runs require initialize_from")
        return self
