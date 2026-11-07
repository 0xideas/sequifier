"""Local trajectory-backed categorical policies and return-conditioned learning."""

from sequifier.config.policy_config import (
    CategoricalPolicyConfig,
    ControlExperimentConfig,
    PolicyTrainingConfig,
    ReturnDefinition,
)
from sequifier.rl.checkpoint import load_policy, save_policy
from sequifier.rl.distributions import (
    CategoricalActionDistribution,
    masked_action_likelihood,
)
from sequifier.rl.experiment import collect_control_dataset, run_control_experiment
from sequifier.rl.history import PolicyInputs, action_index, build_policy_inputs
from sequifier.rl.policy import CategoricalPolicy
from sequifier.rl.runtime import PolicyExecutor, collect_policy_rollout
from sequifier.rl.training import PolicyTrainingResult, train_policy
from sequifier.trajectory.dataset import (
    TrajectoryDataset,
    collate_policy_examples,
    split_by_episode,
)
from sequifier.trajectory.returns import remaining_return, return_to_go

__all__ = [
    "CategoricalPolicyConfig",
    "ControlExperimentConfig",
    "PolicyTrainingConfig",
    "ReturnDefinition",
    "CategoricalActionDistribution",
    "CategoricalPolicy",
    "PolicyInputs",
    "PolicyExecutor",
    "PolicyTrainingResult",
    "TrajectoryDataset",
    "action_index",
    "build_policy_inputs",
    "collate_policy_examples",
    "collect_control_dataset",
    "collect_policy_rollout",
    "load_policy",
    "masked_action_likelihood",
    "remaining_return",
    "return_to_go",
    "run_control_experiment",
    "save_policy",
    "split_by_episode",
    "train_policy",
]
