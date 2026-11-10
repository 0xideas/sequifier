"""Dispatch built-in phases to their runtimes and data preparation functions."""

from dataclasses import dataclass
from typing import Callable

from sequifier.io.preference_dataset import prepare_preferences
from sequifier.io.prepared_policy_data import PreparedPolicyData
from sequifier.rl.dpo import run_dpo_phase
from sequifier.rl.grpo import run_grpo_phase
from sequifier.rl.iql import run_iql_phase
from sequifier.rl.offline_data import prepare_iql, prepare_offline
from sequifier.rl.ppo import run_ppo_phase
from sequifier.rl.training import run_imitation_phase


@dataclass(frozen=True)
class BuiltinAlgorithm:
    runtime: Callable
    prepare_data: Callable[..., PreparedPolicyData] | None = None


BUILTIN_ALGORITHMS = {
    "policy_imitation": BuiltinAlgorithm(run_imitation_phase, prepare_offline),
    "iql": BuiltinAlgorithm(run_iql_phase, prepare_iql),
    "ppo": BuiltinAlgorithm(run_ppo_phase),
    "dpo": BuiltinAlgorithm(run_dpo_phase, prepare_preferences),
    "grpo": BuiltinAlgorithm(run_grpo_phase),
}
