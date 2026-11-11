"""Policy rollout through the same environment and causal history used in training."""

from copy import deepcopy
from typing import Literal
from uuid import uuid4

from sequifier.config.policy_infer_config import PolicyRolloutConfig
from sequifier.config.rollout_config import InputBinding
from sequifier.io.actions import action_component_key, action_latent_key
from sequifier.io.values import numeric_scalar
from sequifier.rl.collection import PolicyCursor
from sequifier.rl.environment import (
    OnlineEnvironmentAdapter,
    OnlineResetResult,
    require_seeded_reset,
    validate_online_adapter,
    validate_reset,
)
from sequifier.rl.policy import Policy
from sequifier.trajectory.recording import TrajectoryRecorder, close_adapter
from sequifier.trajectory.schema import SeedHistory, Trajectory


def collect_policy_rollout(
    policy: Policy,
    adapter: OnlineEnvironmentAdapter,
    seed: SeedHistory,
    config: PolicyRolloutConfig,
    max_steps: int,
    *,
    desired_return: float | None = None,
    action_selection: Literal["deterministic", "stochastic"] = "deterministic",
    random_seed: int = 1010,
) -> Trajectory:
    """Execute a supplied initial observation, preserving actual episode boundaries."""
    was_training, failed = policy.training, False
    try:
        validate_online_adapter(adapter, config.environment)
        require_seeded_reset(adapter)
        if (not isinstance(max_steps, int)) or max_steps < 1:
            raise ValueError("policy collection budget must be a positive integer")
        if action_selection not in {"deterministic", "stochastic"}:
            raise ValueError("action_selection must be deterministic or stochastic")
        spec, units = policy.config, config.environment.units
        if units != getattr(policy, "units", {}):
            raise ValueError("rollout environment units differ from the policy")
        seed = SeedHistory.model_validate(seed.model_dump())
        if (
            sum(seed.attention_valid_mask) != 1
            or len(seed.attention_valid_mask) != spec.context_length
        ):
            raise ValueError(
                "policy seed requires one current observation in its context window"
            )
        if not spec.return_conditioned and desired_return is not None:
            raise ValueError("behavior cloning has no desired-return input")
        if spec.return_conditioned and desired_return is None:
            desired_return = spec.initial_desired_return
        returns = (
            [numeric_scalar(desired_return, "desired_return")]
            if spec.return_conditioned
            else []
        )
        if (
            spec.return_conditioned
            and spec.returns.finite_horizon is not None
            and config.max_episode_steps != spec.returns.finite_horizon
        ):
            raise ValueError(
                "policy max_episode_steps must equal its finite return horizon"
            )
        initial = OnlineResetResult(
            {column: values[-1] for column, values in seed.observations.items()},
            seed.static_inputs,
            units,
        )
        episode_id = str(uuid4())
        policy.eval()
        cursor = PolicyCursor(
            policy, adapter, units, random_seed, strict_vocabulary=False
        )
        cursor.reset(initial, episode_id, returns[0] if returns else None)
        reset = validate_reset(
            adapter.reset_from_observation(deepcopy(initial), random_seed), units
        )
        if dict(reset.observation) != dict(initial.observation) or dict(
            reset.static_inputs
        ) != dict(initial.static_inputs):
            raise ValueError(
                "seeded reset must preserve the supplied observation and static inputs"
            )
        recorder = TrajectoryRecorder(episode_id, config.max_episode_steps)
        previous_actions, log_probabilities = [], []
        for step_id in range(max_steps):
            current = cursor.history.current
            decision = cursor.decide(deterministic=action_selection == "deterministic")
            result = cursor.step(decision)

            previous_actions.append(current.previous_action)
            log_probabilities.append(decision.log_probability)
            if spec.return_conditioned:
                assert cursor.desired_return is not None
                returns.append(cursor.desired_return)
            successor = (
                dict(result.observation) if result.observation is not None else None
            )
            action_metadata = (
                decision.action if isinstance(decision.action, dict) else {}
            )
            transition = recorder.append(
                observation=dict(current.observation),
                predictions=decision.domain_action,
                action_latents={
                    name: action_metadata[action_latent_key(name, spec)]
                    for name, field in spec.actions.items()
                    if field.type == "real"
                },
                action_components={
                    name: action_metadata[action_component_key(name, spec)]
                    for name, field in spec.actions.items()
                    if field.type == "real"
                },
                external_inputs=successor or {},
                reward=result.reward,
                successor_observation=successor,
                terminated=result.terminated,
                truncated=result.truncated,
            )
            if transition.terminated or transition.truncated:
                break
        return recorder.finish(
            format="sequifier.policy-trajectory",
            version=2,
            seed=seed,
            bindings=[
                InputBinding(input=name, source="environment", key=name)
                for name in spec.observation_columns
            ],
            prediction_columns=spec.action_columns,
            metadata={
                "policy_execution": {
                    "config": spec.model_dump(mode="json"),
                    "initial_desired_return": desired_return,
                    "remaining_returns": returns,
                    "previous_action_ids": previous_actions,
                    "action_log_probabilities": log_probabilities,
                    "action_selection": action_selection,
                },
            },
        )
    except BaseException:
        failed = True
        raise
    finally:
        policy.train(was_training)
        close_adapter(adapter, failed)
