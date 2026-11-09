"""One explicit built-in inventory for phase configuration and execution."""

from dataclasses import dataclass
from typing import Callable, get_args

from sequifier.config.policy_plan import (
    DPOPhaseConfig,
    GRPOPhaseConfig,
    IQLPhaseConfig,
    PolicyImitationPhaseConfig,
    PolicyPhase,
    PPOPhaseConfig,
)
from sequifier.io.preference_dataset import prepare_preferences
from sequifier.rl.dpo import run_dpo_phase
from sequifier.rl.grpo import run_grpo_phase
from sequifier.rl.iql import run_iql_phase
from sequifier.rl.offline_data import prepare_offline
from sequifier.rl.ppo import run_ppo_phase
from sequifier.rl.training import run_imitation_phase


@dataclass(frozen=True)
class BuiltinAlgorithm:
    config_type: type
    runtime: Callable
    prepare_data: Callable | None = None
    state_version: int = 1


BUILTIN_ALGORITHMS = {
    "policy_imitation": BuiltinAlgorithm(
        PolicyImitationPhaseConfig, run_imitation_phase, prepare_offline
    ),
    "iql": BuiltinAlgorithm(IQLPhaseConfig, run_iql_phase, prepare_offline),
    "ppo": BuiltinAlgorithm(PPOPhaseConfig, run_ppo_phase),
    "dpo": BuiltinAlgorithm(DPOPhaseConfig, run_dpo_phase, prepare_preferences),
    "grpo": BuiltinAlgorithm(GRPOPhaseConfig, run_grpo_phase),
}


def validate_registry():
    union_types = set(get_args(get_args(PolicyPhase)[0]))
    registered = {entry.config_type for entry in BUILTIN_ALGORITHMS.values()}
    if union_types != registered or any(
        get_args(entry.config_type.model_fields["algorithm"].annotation) != (name,)
        for name, entry in BUILTIN_ALGORITHMS.items()
    ):
        raise RuntimeError(
            "built-in policy algorithm registry and discriminated union disagree"
        )
