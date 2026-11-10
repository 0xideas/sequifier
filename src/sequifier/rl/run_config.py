"""Resolved inputs for the native policy runner; never an authored config."""

from pydantic import BaseModel, ConfigDict

from sequifier.config.components import ResumeConfig
from sequifier.config.policy_config import ResolvedPolicyBinding
from sequifier.config.policy_plan import FrozenNormalizationConfig, PolicyPhase
from sequifier.config.policy_train_config import SequenceSourceConfig
from sequifier.config.train_config import ModelSpecModel


class ResolvedPolicyRun(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    project_root: str
    model_name: str
    device: str
    seed: int
    enforce_determinism: bool = False
    model: ModelSpecModel
    policy: ResolvedPolicyBinding
    execution_model: dict
    policy_interface: str
    initialize_from: str | None = None
    normalization: FrozenNormalizationConfig | None = None
    sources: dict[str, SequenceSourceConfig]
    preferences: dict[str, str]
    units: dict[str, str]
    phases: list[PolicyPhase]
    validation_fraction: float
    episode_cache_size: int
    resume: ResumeConfig
    model_path: str
    report_path: str
    checkpoint_path: str
    phase_directory: str
