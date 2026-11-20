"""Dispatch built-in phases to their runtimes and data preparation functions."""

from dataclasses import dataclass
from typing import Callable

from sequifier.io.preference_dataset import prepare_preferences
from sequifier.io.prepared_policy_data import PreparedPolicyData
from sequifier.rl.dpo import run_dpo_phase
from sequifier.rl.grpo import run_grpo_phase, validate_grpo_adapter
from sequifier.rl.iql import build_iql_roles, run_iql_phase
from sequifier.rl.offline_data import prepare_iql, prepare_offline
from sequifier.rl.ppo import build_ppo_roles, run_ppo_phase, validate_ppo_adapter
from sequifier.rl.training import run_imitation_phase


@dataclass(frozen=True)
class BuiltinAlgorithm:
    runtime: Callable
    prepare_data: Callable[..., PreparedPolicyData] | None = None
    prepare_for_fitting: Callable[..., PreparedPolicyData] | None = None
    build_roles: Callable | None = None
    validate_adapter: Callable | None = None


BUILTIN_ALGORITHMS = {
    "policy_imitation": BuiltinAlgorithm(run_imitation_phase, prepare_offline),
    "iql": BuiltinAlgorithm(
        run_iql_phase,
        prepare_iql,
        prepare_for_fitting=prepare_offline,
        build_roles=build_iql_roles,
    ),
    "ppo": BuiltinAlgorithm(
        run_ppo_phase,
        build_roles=build_ppo_roles,
        validate_adapter=validate_ppo_adapter,
    ),
    "dpo": BuiltinAlgorithm(run_dpo_phase, prepare_preferences),
    "grpo": BuiltinAlgorithm(run_grpo_phase, validate_adapter=validate_grpo_adapter),
}
