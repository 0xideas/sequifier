"""Resolved inputs for the native policy runner; never an authored config."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from sequifier.config.components import ResumeConfig
from sequifier.config.normalization_config import FrozenNormalizationConfig
from sequifier.config.policy_config import ResolvedPolicyBinding
from sequifier.config.policy_plan import PolicyPhase
from sequifier.config.policy_train_config import SequenceSourceConfig
from sequifier.config.train_config import ModelSpecModel


class ResolvedPolicyRun(BaseModel):
    _prepared_inputs: Any = PrivateAttr(default=None)

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
    latest_checkpoint_interval_minutes: float | None = None
    source_validation_fractions: dict[str, float] = Field(default_factory=dict)
    validation_fraction: float
    episode_cache_size: int
    resume: ResumeConfig
    embedding_layer_names: list[str] = Field(
        default_factory=lambda: ["backbone.final_norm"]
    )
    model_path: str
    report_path: str
    checkpoint_path: str
    phase_directory: str
