"""Policy rollout through the same environment and causal history used in training."""

import math
import warnings
from copy import deepcopy
from uuid import uuid4

from sequifier.config.policy_infer_config import PolicyRolloutConfig
from sequifier.config.rollout_config import InputBinding
from sequifier.io.values import numeric_scalar
from sequifier.rl.collection import decide, validate_online_inputs
from sequifier.rl.distributions import action_generator
from sequifier.rl.environment import (
    OnlineEnvironmentAdapter,
    OnlineResetResult,
    require_seeded_reset,
    validate_online_adapter,
    validate_reset,
    validate_step,
)
from sequifier.rl.history import PolicyHistory
from sequifier.rl.policy import CategoricalPolicy
from sequifier.trajectory.returns import remaining_return
from sequifier.trajectory.schema import SeedHistory, Trajectory, Transition


def collect_policy_rollout(
    policy: CategoricalPolicy,
    adapter: OnlineEnvironmentAdapter,
    seed: SeedHistory,
    config: PolicyRolloutConfig,
    max_steps: int,
    *,
    desired_return: float | None = None,
    deterministic: bool = True,
    random_seed: int = 1010,
) -> Trajectory:
    """Execute a supplied initial observation, preserving actual episode boundaries."""
    was_training, failed = policy.training, False
    try:
        validate_online_adapter(adapter, config.environment)
        require_seeded_reset(adapter)
        if (not isinstance(max_steps, int)) or max_steps < 1:
            raise ValueError("policy collection budget must be a positive integer")
        if not isinstance(deterministic, bool):
            raise ValueError("policy deterministic selection must be a boolean")
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
        history = PolicyHistory(spec)
        history.reset(
            initial.observation,
            initial.static_inputs,
            episode_id,
            returns[0] if returns else None,
        )
        policy.eval()
        validate_online_inputs(policy, history.inputs())
        reset = validate_reset(
            adapter.reset_from_observation(deepcopy(initial), random_seed), units
        )
        if dict(reset.observation) != dict(initial.observation) or dict(
            reset.static_inputs
        ) != dict(initial.static_inputs):
            raise ValueError(
                "seeded reset must preserve the supplied observation and static inputs"
            )
        generator = action_generator(policy.device, random_seed)
        transitions, previous_actions, log_probabilities = [], [], []
        for step_id in range(max_steps):
            current = history.current
            decision = decide(
                policy, None, history.inputs(), generator, deterministic=deterministic
            )
            result = validate_step(adapter.step(decision.domain_action), units)
            truncated = result.truncated or (
                config.max_episode_steps is not None
                and step_id + 1 >= config.max_episode_steps
            )
            previous_actions.append(current.previous_action)
            log_probabilities.append(decision.log_probability)
            if spec.return_conditioned:
                returns.append(
                    remaining_return(returns[-1], result.reward, spec.returns.gamma)
                )
            successor = (
                dict(result.observation) if result.observation is not None else None
            )
            if successor is not None:
                history.advance(
                    decision.action, successor, returns[-1] if returns else None
                )
                validate_online_inputs(policy, history.inputs())
            transitions.append(
                Transition(
                    episode_id=episode_id,
                    step_id=step_id,
                    observation=dict(current.observation),
                    predictions=decision.domain_action,
                    external_inputs=successor or {},
                    reward=result.reward,
                    successor_observation=successor,
                    terminated=result.terminated,
                    truncated=truncated,
                    bootstrap_allowed=not result.terminated,
                )
            )
            if result.terminated or truncated:
                break
        last = transitions[-1]
        return Trajectory(
            format="sequifier.scalar-policy-trajectory",
            episode_id=episode_id,
            seed=seed,
            bindings=[
                InputBinding(input=name, source="environment", key=name)
                for name in spec.observation_columns
            ],
            prediction_columns=[spec.action_column],
            transitions=transitions,
            collection_cutoff=not (last.terminated or last.truncated),
            total_return=math.fsum(step.reward for step in transitions),
            metadata={
                "policy_alignment": spec.alignment,
                "policy_execution": {
                    "config": spec.model_dump(mode="json"),
                    "initial_desired_return": desired_return,
                    "remaining_returns": returns,
                    "previous_action_ids": previous_actions,
                    "action_log_probabilities": log_probabilities,
                    "deterministic": deterministic,
                },
            },
        )
    except BaseException:
        failed = True
        raise
    finally:
        policy.train(was_training)
        try:
            adapter.close()
        except Exception as exc:
            if not failed:
                raise
            warnings.warn(f"Adapter close also failed: {exc}", RuntimeWarning)
