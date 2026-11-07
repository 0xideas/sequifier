"""Return-conditioned actors executed through the ordinary rollout coordinator."""

from collections import deque

import torch

from sequifier.config.rollout_config import RolloutConfig
from sequifier.io.values import DomainValue, domain_fields
from sequifier.rl.distributions import CategoricalActionDistribution
from sequifier.rl.history import build_policy_inputs
from sequifier.rl.policy import CategoricalPolicy
from sequifier.rollout.contracts import EnvironmentAdapter, ResetRequest
from sequifier.rollout.coordinator import collect_rollout
from sequifier.rollout.history import numeric_scalar
from sequifier.trajectory.returns import remaining_return
from sequifier.trajectory.schema import (
    SeedHistory,
    Trajectory,
    Transition,
    validate_trajectory,
)


class PolicyExecutor:
    """Callable predictor/observer; current actions enter history only after step()."""

    def __init__(
        self,
        policy: CategoricalPolicy,
        desired_return: float | None,
        deterministic: bool = True,
    ):
        self.policy = policy
        self.initial_return = desired_return
        self.deterministic = deterministic
        self.active = False

    def reset(self, request: ResetRequest) -> None:
        spec = self.policy.config
        if not isinstance(self.deterministic, bool):
            raise ValueError("policy deterministic selection must be a boolean")
        if sum(request.attention_valid_mask) != 1:
            raise ValueError(
                "policy reset requires one current observation; prior actions are unknown"
            )
        if spec.return_conditioned:
            self.budget = numeric_scalar(self.initial_return, "initial desired_return")
        else:
            if self.initial_return is not None:
                raise ValueError("behavior cloning has no desired-return input")
            self.budget = None
        self.observations = deque(maxlen=spec.context_length)
        self.previous_actions = deque(maxlen=spec.context_length)
        self.budgets = deque(maxlen=spec.context_length)
        self.previous_action = len(spec.action_values)
        self.return_trace = [self.budget] if spec.return_conditioned else []
        self.previous_action_trace = []
        self.log_probabilities = []
        self.generator = torch.Generator(
            device=self.policy.device if self.policy.device.type == "cuda" else "cpu"
        ).manual_seed(request.seed)
        self.pending = None
        self.ended = False
        self.active = True
        self.episode_id = request.episode_id
        self.step_id = 0

    def __call__(self, history: SeedHistory) -> dict[str, DomainValue]:
        if not self.active or self.pending is not None or self.ended:
            raise ValueError(
                "policy predictions require an active episode and an executed previous prediction"
            )
        spec = self.policy.config
        observation = {
            column: values[-1] for column, values in history.observations.items()
        }
        inputs = build_policy_inputs(
            spec,
            list(self.observations) + [observation],
            list(self.previous_actions) + [self.previous_action],
            list(self.budgets) + [self.budget] if spec.return_conditioned else None,
            history.static_inputs,
        )
        with torch.no_grad():
            distribution = CategoricalActionDistribution(
                self.policy(inputs.to(self.policy.device))[:, -1, :]
            )
            action = (
                distribution.mode()
                if self.deterministic
                else distribution.sample(self.generator)
            )
            log_probability = float(distribution.log_prob(action)[0])
        index = int(action[0])
        self.pending = (observation, index, log_probability)
        return {spec.action_column: spec.action_values[index]}

    def observe(self, transition: Transition) -> None:
        if not self.active or self.pending is None:
            raise ValueError("policy observed a transition without a predicted action")
        observation, index, log_probability = self.pending
        spec = self.policy.config
        if (
            transition.episode_id != self.episode_id
            or transition.step_id != self.step_id
        ):
            raise ValueError("policy transition episode/step IDs are out of sync")
        if transition.observation != observation or transition.predictions != {
            spec.action_column: spec.action_values[index]
        }:
            raise ValueError("executed transition does not match the policy decision")
        self.observations.append(observation)
        self.previous_actions.append(self.previous_action)
        self.previous_action_trace.append(self.previous_action)
        self.log_probabilities.append(log_probability)
        if spec.return_conditioned:
            assert self.budget is not None
            self.budgets.append(self.budget)
            self.budget = remaining_return(
                self.budget, transition.reward, spec.returns.gamma
            )
            self.return_trace.append(self.budget)
        self.previous_action = index
        self.pending = None
        self.ended = transition.terminated or transition.truncated
        self.step_id += 1


def collect_policy_rollout(
    policy: CategoricalPolicy,
    adapter: EnvironmentAdapter,
    seed: SeedHistory,
    config: RolloutConfig,
    max_steps: int,
    *,
    desired_return: float | None = None,
    deterministic: bool = True,
    random_seed: int = 1010,
) -> Trajectory:
    """Collect one reset-initialized policy episode/fragment and always close adapter.

    Each call/reset starts the requested budget again. Time limits and collection
    cutoffs retain the coordinator's distinct boundary semantics.
    """
    executor = PolicyExecutor(policy, desired_return, deterministic)
    spec = policy.config

    def validate_history(history):
        if config.mode != "policy":
            raise ValueError("categorical execution requires rollout mode: policy")
        horizon = spec.returns.finite_horizon if spec.return_conditioned else None
        if horizon is not None and config.max_episode_steps != horizon:
            raise ValueError(
                "finite-horizon policy execution requires max_episode_steps equal to its return horizon"
            )
        observation = {
            column: values[-1] for column, values in history.observations.items()
        }
        domain_fields(observation, set(spec.observation_columns), "policy observation")
        # Validate the final successor too, even when no subsequent action is needed.
        inputs = build_policy_inputs(
            spec,
            [observation],
            [len(spec.action_values)],
            [numeric_scalar(desired_return, "desired_return")]
            if spec.return_conditioned
            else None,
            history.static_inputs,
        )
        policy.validate_inputs(inputs.to(policy.device))

    was_training = policy.training
    policy.eval()
    try:
        trajectory = collect_rollout(
            executor,
            adapter,
            seed,
            config,
            [spec.action_column],
            max_steps,
            random_seed=random_seed,
            history_validator=validate_history,
            observer=executor,
            metadata={"policy_alignment": spec.alignment},
        )
    finally:
        policy.train(was_training)
    trajectory.metadata["policy_execution"] = {
        "config": spec.model_dump(mode="json"),
        "initial_desired_return": desired_return,
        "remaining_returns": executor.return_trace,
        "previous_action_ids": executor.previous_action_trace,
        "action_log_probabilities": executor.log_probabilities,
        "deterministic": deterministic,
    }
    return validate_trajectory(trajectory)
