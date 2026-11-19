"""Shared prediction/execution ordering and single-environment collection."""

from collections.abc import Callable
from copy import deepcopy
from uuid import uuid4

from sequifier.config.rollout_config import RolloutConfig
from sequifier.io.values import DomainValue
from sequifier.rollout.contracts import (
    EnvironmentAdapter,
    ResetRequest,
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
from sequifier.trajectory.recording import TrajectoryRecorder, close_adapter
from sequifier.trajectory.schema import SeedHistory, Trajectory


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
) -> Trajectory:
    """Collect one episode fragment. This call owns and always closes the adapter.

    Predictor receives a copy of the current SeedHistory in original units and
    returns named scalar forecasts. Conversion to model units belongs in that
    predictor wrapper, using fixed training metadata.
    """
    failed = False
    try:
        if (not isinstance(max_steps, int)) or max_steps < 1:
            raise ValueError("rollout collection budget must be a positive integer")
        validate_adapter(adapter)
        history = validate_seed(seed)
        for column, values in history.observations.items():
            for value in values:
                numeric_scalar(value, f"forecast seed.{column}")
        numeric_fields(
            history.static_inputs,
            set(history.static_inputs),
            "forecast static inputs",
        )
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
        recorder = TrajectoryRecorder(episode_id, config.max_episode_steps)
        external_keys = {
            binding.key
            for binding in config.bindings
            if binding.source == "environment"
        }

        for step_id in range(max_steps):
            predictions = numeric_fields(
                predictor(history.model_copy(deep=True)),
                set(prediction_columns),
                "predictions",
            )
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
            external: dict[str, DomainValue] = {
                key: value
                for key, value in numeric_fields(
                    result.external_inputs, external_keys, "adapter.external_inputs"
                ).items()
            }
            reward = numeric_scalar(result.reward, "adapter.reward")
            successor = {
                binding.input: (
                    predictions if binding.source == "prediction" else external
                )[binding.key]
                for binding in config.bindings
            }

            next_history = advance_history(history, successor)
            if history_validator is not None:
                history_validator(next_history.model_copy(deep=True))
            transition = recorder.append(
                observation=observation,
                predictions=predictions,
                external_inputs=external,
                reward=reward,
                successor_observation=successor,
                terminated=result.terminated,
                truncated=result.truncated,
            )
            history = next_history
            if transition.terminated or transition.truncated:
                break
        return recorder.finish(
            format="sequifier.numeric-forecast-trajectory",
            seed=initial_history,
            bindings=config.bindings,
            prediction_columns=prediction_columns,
            metadata=metadata or {},
        )
    except BaseException:
        failed = True
        raise
    finally:
        close_adapter(adapter, failed)
