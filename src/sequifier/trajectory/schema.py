"""Inspectable version-1 forecast and policy collection records."""

import json
import math
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    JsonValue,
    StrictInt,
    TypeAdapter,
    model_validator,
)

from sequifier.config.rollout_config import InputBinding
from sequifier.io.values import DomainValue
from sequifier.trajectory.returns import remaining_return


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SeedHistory(Record):
    observations: dict[str, list[JsonValue]]
    attention_valid_mask: list[bool]
    static_inputs: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_window(self):
        mask = self.attention_valid_mask
        if (
            not mask
            or not mask[-1]
            or any(mask[i] and not mask[i + 1] for i in range(len(mask) - 1))
        ):
            raise ValueError("seed mask must be left padding followed by valid history")
        if not self.observations or any(
            len(values) != len(mask) for values in self.observations.values()
        ):
            raise ValueError(
                "seed feature histories must be nonempty and match the attention mask length"
            )
        if set(self.observations) & set(self.static_inputs):
            raise ValueError("static inputs must be separate from changing inputs")
        return self


class Transition(Record):
    episode_id: str
    step_id: int = Field(ge=0)
    observation: dict[str, JsonValue]
    predictions: dict[str, DomainValue]
    external_inputs: dict[str, JsonValue]
    reward: FiniteFloat
    action_latents: dict[str, FiniteFloat] = Field(default_factory=dict)
    action_components: dict[str, StrictInt] = Field(default_factory=dict)
    successor_observation: dict[str, JsonValue] | None
    terminated: bool
    truncated: bool
    bootstrap_allowed: bool


class Trajectory(Record):
    format: Literal[
        "sequifier.numeric-forecast-trajectory",
        "sequifier.scalar-policy-trajectory",
        "sequifier.policy-trajectory",
    ] = "sequifier.numeric-forecast-trajectory"
    version: Literal[1, 2] = 1
    episode_id: str = Field(min_length=1)
    seed: SeedHistory
    bindings: list[InputBinding]
    prediction_columns: list[str]
    transitions: list[Transition] = Field(min_length=1)
    collection_cutoff: bool
    total_return: FiniteFloat
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_temporal_contract(self):
        # Metadata must round-trip through the same portable JSON representation.
        json.dumps(self.metadata, allow_nan=False)
        inputs = set(self.seed.observations)
        if not self.prediction_columns or len(self.prediction_columns) != len(
            set(self.prediction_columns)
        ):
            raise ValueError("prediction columns must be nonempty and unique")
        bound = [binding.input for binding in self.bindings]
        if len(bound) != len(set(bound)) or set(bound) != inputs:
            raise ValueError(
                "trajectory requires exactly one producer per changing input"
            )
        external_keys = {
            binding.key for binding in self.bindings if binding.source == "environment"
        }
        observation = {
            key: values[-1] for key, values in self.seed.observations.items()
        }
        for index, transition in enumerate(self.transitions):
            if transition.episode_id != self.episode_id or transition.step_id != index:
                raise ValueError("transition episode/step IDs are not contiguous")
            if transition.observation != observation:
                raise ValueError(
                    f"transition {index} observation breaks trajectory continuity"
                )
            if set(transition.predictions) != set(self.prediction_columns):
                raise ValueError(
                    f"transition {index} prediction fields do not match the contract"
                )
            if transition.successor_observation is None:
                if (
                    self.format
                    not in {
                        "sequifier.scalar-policy-trajectory",
                        "sequifier.policy-trajectory",
                    }
                    or not transition.terminated
                    or transition.external_inputs
                ):
                    raise ValueError(
                        "only a terminal policy transition may omit its successor"
                    )
            else:
                if set(transition.external_inputs) != external_keys:
                    raise ValueError(
                        f"transition {index} external input fields do not match bindings"
                    )
                if set(transition.successor_observation) != inputs:
                    raise ValueError(
                        f"transition {index} successor fields do not match inputs"
                    )
                for binding in self.bindings:
                    producer = (
                        transition.predictions
                        if binding.source == "prediction"
                        else transition.external_inputs
                    )
                    if (
                        binding.key not in producer
                        or transition.successor_observation[binding.input]
                        != producer[binding.key]
                    ):
                        raise ValueError(
                            f"transition {index} successor violates binding for {binding.input!r}"
                        )
            if transition.bootstrap_allowed != (not transition.terminated):
                raise ValueError(
                    "termination must disable bootstrap; truncation preserves the final observation"
                )
            if (transition.terminated or transition.truncated) and index != len(
                self.transitions
            ) - 1:
                raise ValueError("trajectory contains a step after an episode boundary")
            observation = transition.successor_observation
        last = self.transitions[-1]
        if self.collection_cutoff != (not (last.terminated or last.truncated)):
            raise ValueError("collection cutoff must leave the episode unfinished")
        expected_return = math.fsum(
            transition.reward for transition in self.transitions
        )
        if not math.isclose(
            self.total_return, expected_return, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("total_return does not equal the sum of rewards")
        if self.format == "sequifier.policy-trajectory":
            if self.version != 2:
                raise ValueError("joint policy trajectories require version 2")
            _validate_joint_execution(self)
        if self.format == "sequifier.scalar-policy-trajectory":
            if (
                sum(self.seed.attention_valid_mask) != 1
                or len(self.prediction_columns) != 1
            ):
                raise ValueError(
                    "policy episodes require one initial observation and one action head"
                )
            if "policy_execution" in self.metadata:
                _validate_policy_execution(self)
        return self


def _validate_policy_execution(trajectory: Trajectory) -> None:
    from sequifier.config.policy_config import CategoricalPolicyConfig

    execution = trajectory.metadata["policy_execution"]
    if not isinstance(execution, dict):
        raise ValueError("policy_execution must be an object")
    if not isinstance(execution.get("deterministic"), bool):
        raise ValueError(
            "policy execution selection must declare a deterministic boolean"
        )
    spec = CategoricalPolicyConfig.model_validate(execution.get("config"))
    if (
        trajectory.prediction_columns != [spec.action_column]
        or set(trajectory.seed.observations) != set(spec.observation_columns)
        or set(trajectory.seed.static_inputs) != set(spec.static_columns)
    ):
        raise ValueError("policy execution schema does not match trajectory fields")
    budgets = TypeAdapter(list[FiniteFloat]).validate_python(
        execution.get("remaining_returns"), strict=True
    )
    previous_actions = TypeAdapter(list[StrictInt]).validate_python(
        execution.get("previous_action_ids"), strict=True
    )
    log_probabilities = TypeAdapter(list[FiniteFloat]).validate_python(
        execution.get("action_log_probabilities"), strict=True
    )
    count = len(trajectory.transitions)
    if (
        len(previous_actions) != count
        or len(log_probabilities) != count
        or any(value > 1e-6 for value in log_probabilities)
    ):
        raise ValueError(
            "policy execution action IDs/log probabilities must match decision count"
        )
    expected_previous = len(spec.action_values)
    for index, step in enumerate(trajectory.transitions):
        action = step.predictions[spec.action_column]
        if (
            action not in spec.action_values
            or previous_actions[index] != expected_previous
        ):
            raise ValueError(
                "policy previous actions must lag executed categorical actions by one decision"
            )
        expected_previous = spec.action_values.index(action)
    if spec.return_conditioned:
        initial = TypeAdapter(FiniteFloat).validate_python(
            execution.get("initial_desired_return"), strict=True
        )
        if len(budgets) != count + 1 or budgets[0] != initial:
            raise ValueError(
                "policy return budgets must include the initial and every successor budget"
            )
        for index, step in enumerate(trajectory.transitions):
            expected = remaining_return(budgets[index], step.reward, spec.returns.gamma)
            if not math.isclose(
                budgets[index + 1], expected, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    "policy return budgets violate the recorded reward/gamma arithmetic"
                )
    elif budgets or execution.get("initial_desired_return") is not None:
        raise ValueError("behavior cloning must not consume desired-return inputs")


def validate_trajectory(value: Trajectory | dict) -> Trajectory:
    """Revalidate even an existing (possibly mutated) trajectory object."""
    return Trajectory.model_validate(
        value.model_dump() if isinstance(value, Trajectory) else value
    )


def _validate_joint_execution(trajectory):
    from sequifier.config.policy_config import CategoricalPolicyConfig
    from sequifier.io.actions import encode_action

    execution = trajectory.metadata.get("policy_execution")
    if not isinstance(execution, dict):
        raise ValueError("policy trajectories require execution metadata")
    spec = CategoricalPolicyConfig.model_validate(execution.get("config"))
    if trajectory.prediction_columns != spec.action_columns:
        raise ValueError("trajectory action fields differ from the binding")
    if set(trajectory.seed.observations) != set(spec.observation_columns) or set(
        trajectory.seed.static_inputs
    ) != set(spec.static_columns):
        raise ValueError("trajectory inputs differ from the model binding")
    scores = TypeAdapter(list[FiniteFloat]).validate_python(
        execution.get("action_log_probabilities"), strict=True
    )
    previous = execution.get("previous_action_ids")
    if (
        len(scores) != len(trajectory.transitions)
        or not isinstance(previous, list)
        or len(previous) != len(scores)
    ):
        raise ValueError("trajectory scores/history must match decision count")
    expected = None
    for index, step in enumerate(trajectory.transitions):
        real_fields = {
            name for name, field in spec.actions.items() if field.type == "real"
        }
        if (set(step.action_latents) | set(step.action_components)) - real_fields:
            raise ValueError(
                "trajectory transform metadata must name real action fields"
            )
        recorded_previous = previous[index]
        if isinstance(recorded_previous, dict):
            recorded_previous = {
                key: value
                for key, value in recorded_previous.items()
                if key in spec.actions
            }
        if recorded_previous != expected:
            raise ValueError("previous actions must lag executed actions")
        expected = encode_action(step.predictions, spec)
    if not isinstance(execution.get("deterministic"), bool):
        raise ValueError("trajectory must declare its selection mode")
    budgets = TypeAdapter(list[FiniteFloat]).validate_python(
        execution.get("remaining_returns"), strict=True
    )
    if spec.return_conditioned:
        if len(budgets) != len(scores) + 1 or budgets[0] != execution.get(
            "initial_desired_return"
        ):
            raise ValueError(
                "return budgets must include every successor and the initial budget"
            )
        for index, step in enumerate(trajectory.transitions):
            if not math.isclose(
                budgets[index + 1],
                remaining_return(budgets[index], step.reward, spec.returns.gamma),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "return budgets violate the recorded reward arithmetic"
                )
    elif budgets:
        raise ValueError("an unconditioned policy cannot record return budgets")
