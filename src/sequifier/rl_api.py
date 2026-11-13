"""Local categorical policy learning, inference, and environment contracts."""

from sequifier.config.normalization_config import FrozenNormalizationConfig
from sequifier.config.parameter_selection import (
    ParameterSelector,
    RoleTrainability,
    TrainabilityConfig,
)
from sequifier.config.phase_controls import Cadence, StoppingConfig
from sequifier.config.policy_config import (
    ActionFieldConfig,
    PolicyBinding,
    ResolvedPolicyBinding,
    ReturnDefinition,
)
from sequifier.config.policy_infer_config import (
    PolicyInferenceConfig,
    PolicyRolloutConfig,
)
from sequifier.config.policy_optimization import OptimizationConfig
from sequifier.config.policy_plan import (
    DPOPhaseConfig,
    GRPOEvaluationConfig,
    GRPOPhaseConfig,
    IQLPhaseConfig,
    OnlineEnvironmentConfig,
    OnlineEvaluationConfig,
    PolicyImitationPhaseConfig,
    PPOPhaseConfig,
    ReferenceConfig,
)
from sequifier.config.policy_train_config import (
    PolicyTableColumns,
    SequenceSourceConfig,
)
from sequifier.config.sequence_import_config import SequenceImportConfig
from sequifier.config.train_config import ResolvedSequifierConfig, SequifierConfig
from sequifier.io.batch import SequifierBatch, SequifierInputs, collate_sequence_batches
from sequifier.io.policy_dataset import PolicySequenceDataset, split_by_episode
from sequifier.io.policy_window import PolicyWindowView, action_index
from sequifier.io.sequence import SequenceEpisode, TransitionColumns
from sequifier.rl.checkpoint import load_policy, save_policy
from sequifier.rl.distributions import (
    CategoricalActionDistribution,
    masked_action_nll_loss,
)
from sequifier.rl.environment import (
    EnvironmentCapabilities,
    OnlineEnvironmentAdapter,
    OnlineResetResult,
    OnlineStepResult,
)
from sequifier.rl.learner import run_policy_training
from sequifier.rl.policy import Policy
from sequifier.rl.ppo_buffer import load_collection_record
from sequifier.rl.ppo_evaluation import evaluate_policy
from sequifier.rl.run_state import (
    ActorRepresentation,
    PolicyPhaseResult,
    PolicyRunResult,
)
from sequifier.rl.runtime import collect_policy_rollout
from sequifier.trajectory.export import import_sequences
from sequifier.trajectory.importers import source_episodes
from sequifier.trajectory.preferences import (
    PreferenceManifest,
    PreferencePair,
    PreferenceSpan,
    load_preferences,
)
from sequifier.trajectory.returns import remaining_return, return_to_go

__all__ = [
    "ParameterSelector",
    "RoleTrainability",
    "TrainabilityConfig",
    "Cadence",
    "StoppingConfig",
    "ActorRepresentation",
    "CategoricalActionDistribution",
    "Policy",
    "ResolvedPolicyBinding",
    "DPOPhaseConfig",
    "EnvironmentCapabilities",
    "FrozenNormalizationConfig",
    "GRPOPhaseConfig",
    "GRPOEvaluationConfig",
    "IQLPhaseConfig",
    "OnlineEnvironmentAdapter",
    "OnlineEnvironmentConfig",
    "OnlineEvaluationConfig",
    "OnlineResetResult",
    "OnlineStepResult",
    "PolicyImitationPhaseConfig",
    "PolicyRolloutConfig",
    "PolicyInferenceConfig",
    "OptimizationConfig",
    "PolicyPhaseResult",
    "SequifierConfig",
    "ResolvedSequifierConfig",
    "PolicyRunResult",
    "ResolvedPolicyBinding",
    "PolicySequenceDataset",
    "PolicyTableColumns",
    "PolicyWindowView",
    "PPOPhaseConfig",
    "PreferenceManifest",
    "PreferencePair",
    "PreferenceSpan",
    "ReferenceConfig",
    "ReturnDefinition",
    "SequenceEpisode",
    "SequenceImportConfig",
    "SequenceSourceConfig",
    "SequifierBatch",
    "SequifierInputs",
    "TransitionColumns",
    "action_index",
    "collate_sequence_batches",
    "collect_policy_rollout",
    "evaluate_policy",
    "import_sequences",
    "load_collection_record",
    "load_policy",
    "load_preferences",
    "masked_action_nll_loss",
    "remaining_return",
    "return_to_go",
    "run_policy_training",
    "save_policy",
    "source_episodes",
    "split_by_episode",
]

__all__ += ["ActionFieldConfig", "Policy", "PolicyBinding"]
