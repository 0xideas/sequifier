"""Local categorical policy learning, online collection and execution."""

from sequifier.config.policy_config import (
    CategoricalPolicyConfig,
    PolicySchemaConfig,
    ReturnDefinition,
)
from sequifier.config.policy_infer_config import PolicyInferenceConfig
from sequifier.config.policy_train_config import (
    PolicyTableColumns,
    SequenceSourceConfig,
)
from sequifier.config.sequence_import_config import SequenceImportConfig
from sequifier.config.train_config import PolicyTrainConfig, PPOTrainConfig
from sequifier.config.training_plan import (
    IQLPhaseConfig,
    OnlineEnvironmentConfig,
    PolicyImitationPhaseConfig,
    PPOEvaluationConfig,
    PPONormalizationConfig,
    PPOPhaseConfig,
    PPOResumeConfig,
)
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
    OnlineEnvironmentAdapter,
    OnlineResetResult,
    OnlineStepResult,
    SeededAdapterBridge,
)
from sequifier.rl.iql import IQLTrainingResult, run_iql_training
from sequifier.rl.learner import run_policy_training
from sequifier.rl.policy import CategoricalPolicy
from sequifier.rl.ppo import PPOTrainingResult, run_ppo_training
from sequifier.rl.ppo_buffer import load_collection_record
from sequifier.rl.ppo_evaluation import evaluate_policy
from sequifier.rl.runtime import PolicyExecutor, collect_policy_rollout
from sequifier.rl.training import PolicyTrainingResult, train_policy
from sequifier.trajectory.importers import import_sequences, source_episodes
from sequifier.trajectory.returns import remaining_return, return_to_go

__all__ = [
    "SequifierInputs",
    "SequifierBatch",
    "SequenceEpisode",
    "TransitionColumns",
    "PolicyWindowView",
    "PolicySchemaConfig",
    "PolicySequenceDataset",
    "SequenceSourceConfig",
    "PolicyTableColumns",
    "SequenceImportConfig",
    "source_episodes",
    "import_sequences",
    "PolicyInferenceConfig",
    "PolicyTrainConfig",
    "PPOTrainConfig",
    "PPOPhaseConfig",
    "OnlineEnvironmentConfig",
    "PPOEvaluationConfig",
    "PPONormalizationConfig",
    "PPOResumeConfig",
    "OnlineEnvironmentAdapter",
    "OnlineResetResult",
    "OnlineStepResult",
    "SeededAdapterBridge",
    "PPOTrainingResult",
    "run_ppo_training",
    "evaluate_policy",
    "load_collection_record",
    "PolicyImitationPhaseConfig",
    "IQLPhaseConfig",
    "run_policy_training",
    "run_iql_training",
    "IQLTrainingResult",
    "CategoricalPolicyConfig",
    "ReturnDefinition",
    "CategoricalActionDistribution",
    "CategoricalPolicy",
    "PolicyExecutor",
    "PolicyTrainingResult",
    "action_index",
    "collect_policy_rollout",
    "load_policy",
    "masked_action_nll_loss",
    "remaining_return",
    "return_to_go",
    "save_policy",
    "split_by_episode",
    "collate_sequence_batches",
    "train_policy",
]
