"""Bounded local policy evaluation with independent seeds, adapters and RNG."""

import math
import signal
import statistics
import threading
import time
from contextlib import contextmanager
from copy import deepcopy

from sequifier.rl.collection import decide, stream_seed, validate_online_inputs
from sequifier.rl.distributions import action_generator
from sequifier.rl.environment import load_online_adapter, validate_reset, validate_step
from sequifier.rl.history import PolicyHistory
from sequifier.runtime.random_state import capture_rng_state, restore_rng_state


class EvaluationTimeout(TimeoutError):
    pass


def require_evaluation_timer():
    if threading.current_thread() is not threading.main_thread() or not hasattr(
        signal, "setitimer"
    ):
        raise ValueError(
            "bounded local PPO evaluation requires the main thread and a POSIX interval timer"
        )


@contextmanager
def _deadline(seconds):
    require_evaluation_timer()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expired(signum, frame):
        raise EvaluationTimeout("PPO evaluation exceeded timeout_seconds")

    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(
                signal.ITIMER_REAL,
                max(1e-6, previous_timer[0] - (time.monotonic() - started)),
                previous_timer[1],
            )


def evaluate_policy(actor, evaluation, *, forbidden_adapter=None):
    """Evaluate a frozen snapshot; completed returns exclude censored episodes.

    The timeout covers adapter creation/reset/steps for the entire evaluation.
    Adapters must release resources promptly from close().
    """
    require_evaluation_timer()
    if evaluation.environment.units != getattr(actor, "units", {}):
        raise ValueError(
            "evaluation environment units differ from the portable actor representation"
        )
    saved_random = capture_rng_state(actor.device)
    started = time.monotonic()
    adapter = None
    episodes = []
    timed_out = False
    attempted = 0
    episode_return = 0.0
    length = 0
    active = False
    try:
        snapshot = deepcopy(actor).eval()
        with _deadline(evaluation.timeout_seconds):
            adapter = load_online_adapter(evaluation.environment)
            if adapter is forbidden_adapter:
                adapter = None  # The training owner must retain and close it.
                raise ValueError(
                    "evaluation factory returned the live training adapter"
                )
            for index, seed in enumerate(evaluation.seeds):
                attempted += 1
                episode_return, length = 0.0, 0
                active = True
                result = validate_reset(
                    adapter.reset(stream_seed(seed, "evaluation-reset", 0)),
                    evaluation.environment.units,
                )
                history = PolicyHistory(snapshot.config, strict_vocabulary=True)
                history.reset(
                    result.observation, result.static_inputs, f"evaluation-{index}"
                )
                generator = action_generator(
                    snapshot.device, stream_seed(seed, "evaluation-actions", 0)
                )
                terminated = False
                success = None
                for _ in range(evaluation.max_episode_steps):
                    decision = decide(
                        snapshot,
                        None,
                        history.inputs(),
                        generator,
                        deterministic=evaluation.selection == "deterministic",
                    )
                    step = validate_step(
                        adapter.step(decision.domain_action),
                        evaluation.environment.units,
                    )
                    length += 1
                    episode_return += float(step.reward)
                    if not math.isfinite(episode_return):
                        raise ValueError("non-finite evaluation episode return")
                    if step.observation is not None:
                        history.advance(decision.action, step.observation)
                        validate_online_inputs(snapshot, history.inputs())
                    terminated = step.terminated
                    if step.terminated or step.truncated:
                        success = step.success
                        break
                episodes.append(
                    {
                        "seed": seed,
                        "return": episode_return,
                        "length": length,
                        "completed": terminated,
                        "censored": not terminated,
                        "success": success if terminated else None,
                    }
                )
                active = False
    except EvaluationTimeout:
        timed_out = True
        if active:
            episodes.append(
                {
                    "seed": evaluation.seeds[attempted - 1],
                    "return": episode_return,
                    "length": length,
                    "completed": False,
                    "censored": True,
                    "success": None,
                }
            )
    finally:
        try:
            if adapter is not None:
                adapter.close()
        finally:
            restore_rng_state(actor.device, saved_random)
    completed = [episode for episode in episodes if episode["completed"]]
    returns = [episode["return"] for episode in completed]
    lengths = [episode["length"] for episode in completed]
    successes = [episode["success"] for episode in completed]
    result = {
        "mean_completed_return": statistics.fmean(returns) if returns else None,
        "return_std": statistics.pstdev(returns) if returns else None,
        "return_min": min(returns) if returns else None,
        "return_max": max(returns) if returns else None,
        "mean_completed_length": statistics.fmean(lengths) if lengths else None,
        "mean_length": statistics.fmean(episode["length"] for episode in episodes)
        if episodes
        else None,
        "completed_count": len(completed),
        "censored_count": len(episodes) - len(completed),
        "unattempted_count": len(evaluation.seeds) - attempted,
        "timed_out": timed_out,
        "episodes": episodes,
        "seconds": time.monotonic() - started,
        "selection": evaluation.selection,
    }
    if successes and all(success is not None for success in successes):
        result["success_rate"] = statistics.fmean(successes)
    if evaluation.metric == "score_mean":
        if timed_out or attempted != len(evaluation.seeds):
            raise ValueError("GRPO evaluation requires every finite-horizon score")
        result["score_mean"] = statistics.fmean(
            episode["return"] for episode in episodes
        )
    result["selection_metric"] = evaluation.metric
    result["selection_value"] = result.get(evaluation.metric)
    result["selection_available"] = result["selection_value"] is not None
    return result
