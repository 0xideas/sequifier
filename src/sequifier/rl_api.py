"""Local categorical policy learning, inference, and environment contracts."""

from sequifier.config.policy_config import (
    CategoricalPolicyConfig,
    PolicySchemaConfig,
    ReturnDefinition,
)
from sequifier.config.policy_infer_config import PolicyInferenceConfig
from sequifier.config.policy_plan import (
    DPOPhaseConfig,
    FrozenNormalizationConfig,
    GRPOEvaluationConfig,
    GRPOPhaseConfig,
    IQLPhaseConfig,
    OnlineEnvironmentConfig,
    OnlineEvaluationConfig,
    PolicyImitationPhaseConfig,
    PolicyResumeConfig,
    PPOPhaseConfig,
    ReferenceConfig,
)
from sequifier.config.policy_train_config import (
    PolicyTableColumns,
    SequenceSourceConfig,
)
from sequifier.config.sequence_import_config import SequenceImportConfig
from sequifier.config.train_config import PolicyRunConfig
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
    SeededAdapterBridge,
)
from sequifier.rl.learner import run_policy_training
from sequifier.rl.policy import CategoricalPolicy
from sequifier.rl.ppo_buffer import load_collection_record
from sequifier.rl.ppo_evaluation import evaluate_policy
from sequifier.rl.run_state import (
    ActorRepresentation,
    PolicyPhaseResult,
    PolicyRunResult,
)
from sequifier.rl.runtime import PolicyExecutor, collect_policy_rollout
from sequifier.trajectory.importers import import_sequences, source_episodes
from sequifier.trajectory.preferences import (
    PreferenceManifest,
    PreferencePair,
    PreferenceSpan,
    load_preferences,
)
from sequifier.trajectory.returns import remaining_return, return_to_go

__all__ = [
    "ActorRepresentation",
    "CategoricalActionDistribution",
    "CategoricalPolicy",
    "CategoricalPolicyConfig",
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
    "PolicyExecutor",
    "PolicyImitationPhaseConfig",
    "PolicyInferenceConfig",
    "PolicyPhaseResult",
    "PolicyResumeConfig",
    "PolicyRunConfig",
    "PolicyRunResult",
    "PolicySchemaConfig",
    "PolicySequenceDataset",
    "PolicyTableColumns",
    "PolicyWindowView",
    "PPOPhaseConfig",
    "PreferenceManifest",
    "PreferencePair",
    "PreferenceSpan",
    "ReferenceConfig",
    "ReturnDefinition",
    "SeededAdapterBridge",
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
