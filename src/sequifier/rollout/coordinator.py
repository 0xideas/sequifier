"""Shared prediction/execution ordering and single-environment collection."""

import math
import warnings
from collections.abc import Callable
from copy import deepcopy
from uuid import uuid4

from sequifier.config.rollout_config import RolloutConfig
from sequifier.rollout.contracts import (
    EnvironmentAdapter,
    ResetRequest,
    RolloutObserver,
    StepRequest,
    StepResult,
)
from sequifier.rollout.factories import validate_adapter
from sequifier.rollout.history import (
    advance_history,
    numeric_fields,
    numeric_scalar,
    validate_seed,
)
from sequifier.trajectory.schema import SeedHistory, Trajectory, Transition


def coordinate_steps(predict: Callable, execute: Callable, max_steps: int) -> None:
    """Execute every prediction, including the final budgeted one; stop at a boundary."""
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("rollout collection budget must be a positive integer")
    for step_id in range(max_steps):
        prediction = predict()
        if execute(prediction, step_id):
            break


def collect_rollout(
    predictor: Callable,
    adapter: EnvironmentAdapter,
    seed: SeedHistory,
    config: RolloutConfig,
    prediction_columns: list[str],
    max_steps: int,
    *,
    random_seed: int = 1010,
    metadata: dict | None = None,
    history_validator: Callable | None = None,
    observer: RolloutObserver | None = None,
) -> Trajectory:
    """Collect one episode fragment. This call owns and always closes the adapter.

    Predictor receives a copy of the current SeedHistory in original units and
    returns named scalar forecasts. Conversion to model units belongs in that
    predictor wrapper, using fixed training metadata.
    """
    failed = False
    try:
        validate_adapter(adapter)
        history = validate_seed(seed)
        initial_history = history.model_copy(deep=True)
        if history_validator is not None:
            history_validator(history.model_copy(deep=True))
        if not prediction_columns or len(prediction_columns) != len(
            set(prediction_columns)
        ):
            raise ValueError("prediction_columns must be nonempty and unique")
        config.validate_bindings(list(history.observations), prediction_columns)
        episode_id = str(uuid4())
        reset_request = ResetRequest(
            episode_id,
            {key: tuple(values) for key, values in history.observations.items()},
            tuple(history.attention_valid_mask),
            dict(history.static_inputs),
            random_seed,
        )
        adapter.reset(deepcopy(reset_request))
        if observer is not None:
            observer.reset(deepcopy(reset_request))
        transitions = []
        external_keys = {
            binding.key
            for binding in config.bindings
            if binding.source == "environment"
        }

        def predict():
            return numeric_fields(
                predictor(history.model_copy(deep=True)),
                set(prediction_columns),
                "predictions",
            )

        def execute(predictions, step_id):
            nonlocal history
            observation = {
                key: values[-1] for key, values in history.observations.items()
            }
            result = adapter.step(
                StepRequest(
                    episode_id,
                    step_id,
                    dict(observation),
                    dict(predictions),
                    dict(history.static_inputs),
                )
            )
            if not isinstance(result, StepResult):
                raise ValueError("adapter.step must return StepResult")
            if not isinstance(result.terminated, bool) or not isinstance(
                result.truncated, bool
            ):
                raise ValueError(
                    "adapter termination and truncation flags must be separate booleans"
                )
            external = numeric_fields(
                result.external_inputs, external_keys, "adapter.external_inputs"
            )
            reward = numeric_scalar(result.reward, "adapter.reward")
            successor = {
                binding.input: (
                    predictions if binding.source == "prediction" else external
                )[binding.key]
                for binding in config.bindings
            }
            truncated = result.truncated or (
                config.max_episode_steps is not None
                and step_id + 1 >= config.max_episode_steps
            )
            next_history = advance_history(history, successor)
            if history_validator is not None:
                history_validator(next_history.model_copy(deep=True))
            transitions.append(
                Transition(
                    episode_id=episode_id,
                    step_id=step_id,
                    observation=observation,
                    predictions=predictions,
                    external_inputs=external,
                    reward=reward,
                    successor_observation=successor,
                    terminated=result.terminated,
                    truncated=truncated,
                    bootstrap_allowed=not result.terminated,
                )
            )
            if observer is not None:
                observer.observe(transitions[-1].model_copy(deep=True))
            history = next_history
            return result.terminated or truncated

        coordinate_steps(predict, execute, max_steps)
        last = transitions[-1]
        return Trajectory(
            format=(
                "sequifier.numeric-policy-trajectory"
                if config.mode == "policy"
                else "sequifier.numeric-forecast-trajectory"
            ),
            episode_id=episode_id,
            seed=initial_history,
            bindings=config.bindings,
            prediction_columns=prediction_columns,
            transitions=transitions,
            collection_cutoff=not (last.terminated or last.truncated),
            total_return=math.fsum(transition.reward for transition in transitions),
            metadata=metadata or {},
        )
    except BaseException:
        failed = True
        raise
    finally:
        try:
            adapter.close()
        except Exception as exc:
            if not failed:
                raise
            warnings.warn(f"Adapter close also failed: {exc}", RuntimeWarning)
