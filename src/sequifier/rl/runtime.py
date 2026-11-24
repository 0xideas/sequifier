"""Policy rollout through the same environment and causal history used in training."""

from copy import deepcopy
from dataclasses import asdict
from typing import Literal
from uuid import uuid4

import torch

from sequifier.config.policy_infer_config import PolicyRolloutConfig
from sequifier.config.rollout_config import InputBinding
from sequifier.io.values import numeric_scalar, portable_fields
from sequifier.rl.collection import PolicyCursor, PolicyDecision
from sequifier.rl.environment import (
    OnlineEnvironmentAdapter,
    OnlineResetResult,
    OnlineStepResult,
    require_seeded_reset,
    validate_online_adapter,
    validate_reset,
)
from sequifier.rl.policy import Policy
from sequifier.trajectory.recording import TrajectoryRecorder, close_adapter
from sequifier.trajectory.schema import SeedHistory, Trajectory, validate_policy_seed


def collect_policy_rollout(
    policy: Policy,
    adapter: OnlineEnvironmentAdapter,
    seed: SeedHistory,
    config: PolicyRolloutConfig,
    max_steps: int,
    *,
    desired_return: float | None = None,
    seed_previous_actions: list | None = None,
    seed_returns: list | None = None,
    action_selection: Literal["deterministic", "stochastic"] = "deterministic",
    random_seed: int = 1010,
    journal=None,
) -> Trajectory:
    """Execute a supplied initial observation, preserving actual episode boundaries."""
    was_training, failed = policy.training, False
    try:
        validate_online_adapter(adapter)
        require_seeded_reset(adapter)
        if (not isinstance(max_steps, int)) or max_steps < 1:
            raise ValueError("policy collection budget must be a positive integer")
        if action_selection not in {"deterministic", "stochastic"}:
            raise ValueError("action_selection must be deterministic or stochastic")
        spec, units = policy.config, config.environment.units
        if units != policy.units:
            raise ValueError("rollout environment units differ from the policy")
        seed = SeedHistory.model_validate(seed.model_dump())
        count = sum(seed.attention_valid_mask)
        if len(seed.attention_valid_mask) != spec.context_length:
            raise ValueError("policy seed width must match context_length")
        if seed_previous_actions is None:
            if count > 1 and spec.previous_action_inputs:
                raise ValueError(
                    "Multi-observation policy seeds require lagged action history"
                )
            seed_previous_actions = [None] * count
        if not spec.return_conditioned and desired_return is not None:
            raise ValueError("behavior cloning has no desired-return input")
        if spec.return_conditioned and desired_return is None:
            desired_return = spec.initial_desired_return
        returns = (
            [numeric_scalar(desired_return, "desired_return")]
            if spec.return_conditioned
            else []
        )
        horizon = spec.episode_horizon
        if horizon is not None and config.max_episode_steps not in (None, horizon):
            raise ValueError(
                "policy max_episode_steps must equal its finite return horizon"
            )
        initial = OnlineResetResult(
            {column: values[-1] for column, values in seed.observations.items()},
            seed.static_inputs,
            units,
        )
        episode_id = journal.episode_id if journal else str(uuid4())
        policy.eval()
        cursor = PolicyCursor(
            policy, adapter, units, random_seed, strict_vocabulary=False
        )
        if seed_returns is None:
            seed_returns = [desired_return] if spec.return_conditioned else []
        validate_policy_seed(
            spec,
            seed_previous_actions,
            seed_returns,
            count,
            desired_return,
        )
        earlier = [
            (
                {name: values[i - count] for name, values in seed.observations.items()},
                seed_previous_actions[i],
                seed_returns[i] if spec.return_conditioned else None,
            )
            for i in range(count - 1)
        ]
        cursor.reset(
            initial,
            episode_id,
            desired_return,
            earlier=earlier,
            observation_masks=seed.observation_mask[-count:],
            previous_action=seed_previous_actions[-1],
        )

        def reset_adapter():
            return adapter.reset_from_observation(deepcopy(initial), random_seed)

        reset = (
            journal.execute(
                adapter,
                "reset",
                {"initial": asdict(initial), "seed": random_seed},
                reset_adapter,
                OnlineResetResult,
            )
            if journal
            else reset_adapter()
        )
        reset = validate_reset(reset, units)
        if dict(reset.observation) != dict(initial.observation) or dict(
            reset.static_inputs
        ) != dict(initial.static_inputs):
            raise ValueError(
                "seeded reset must preserve the supplied observation and static inputs"
            )
        recorder = TrajectoryRecorder(
            episode_id, config.max_episode_steps if horizon is None else None
        )
        previous_actions, log_probabilities = [], []
        observation = dict(initial.observation)

        def select():
            decision = cursor.decide(deterministic=action_selection == "deterministic")
            return {
                "decision": asdict(decision),
                "rng": cursor.generator.get_state().tolist(),
            }

        for step_id in range(max_steps):
            current = cursor.history.current
            if journal:
                selected = journal.decision(step_id, select)
                decision = PolicyDecision(**selected["decision"])
                cursor.generator.set_state(
                    torch.tensor(selected["rng"], dtype=torch.uint8)
                )
                response = journal.execute(
                    adapter,
                    step_id,
                    decision.domain_action,
                    lambda: adapter.step(decision.domain_action),
                    OnlineStepResult,
                )
                result = cursor.accept_step(decision, response)
            else:
                decision = cursor.decide(
                    deterministic=action_selection == "deterministic"
                )
                result = cursor.step(decision)

            previous_actions.append(current.previous_action)
            log_probabilities.append(decision.log_probability)
            if spec.return_conditioned:
                assert cursor.desired_return is not None
                returns.append(cursor.desired_return)
            successor = (
                portable_fields(result.observation)
                if result.observation is not None
                else None
            )
            latents, components = policy.action_plan.record_layout.unpack_auxiliary(
                decision.action
            )
            transition = recorder.append(
                observation=observation,
                predictions=decision.domain_action,
                action_latents=latents,
                action_components=components,
                external_inputs=successor or {},
                reward=result.reward,
                successor_observation=successor,
                terminated=result.terminated,
                truncated=result.truncated,
            )
            if transition.terminated or transition.truncated:
                break
            observation = successor
        return recorder.finish(
            format="sequifier.policy-trajectory",
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
                    "seed_previous_action_ids": seed_previous_actions,
                    "seed_returns": seed_returns if spec.return_conditioned else [],
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
