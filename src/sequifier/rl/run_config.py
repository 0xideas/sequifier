"""Phase-local view of a compiled run for native policy training."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from sequifier.config.components import ResumeConfig
from sequifier.config.policy_config import ResolvedPolicyBinding
from sequifier.config.policy_plan import PolicyPhase

if TYPE_CHECKING:
    from sequifier.config.train_config import ResolvedSequifierConfig


@dataclass(frozen=True)
class ResolvedPolicyRun:
    parent: "ResolvedSequifierConfig"
    phases: tuple[PolicyPhase, ...]
    phase_offset: int
    policy: ResolvedPolicyBinding
    initialize_from: str | None
    actor_initialization: Literal["attach", "history_only"]
    resume: ResumeConfig
    checkpoint_path: str

    @property
    def outputs(self):
        return self.parent.outputs

    @property
    def project_root(self):
        return self.parent.project_root

    @property
    def model_name(self):
        return self.parent.model_name
