"""Bounded local policy evaluation with independent seeds, adapters and RNG."""

import signal
import statistics
import threading
import time
from contextlib import contextmanager

from sequifier.rl.collection import PolicyCursor, stream_seed
from sequifier.rl.environment import load_online_adapter
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
    """Evaluate synchronously with stochastic layers disabled; completed returns exclude censored episodes.

    The timeout covers adapter creation/reset/steps for the entire evaluation.
    Adapters must release resources promptly from close().
    """
    require_evaluation_timer()
    if evaluation.environment.units != getattr(actor, "units", {}):
        raise ValueError(
            "evaluation environment units differ from the portable actor representation"
        )
    saved_random = capture_rng_state(actor.device)
    training_modes = {module: module.training for module in actor.modules()}
    started = time.monotonic()
    adapter = None
    episodes = []
    timed_out = False
    try:
        actor.eval()
        with _deadline(evaluation.timeout_seconds):
            adapter = load_online_adapter(evaluation.environment)
            if adapter is forbidden_adapter:
                adapter = None  # The training owner must retain and close it.
                raise ValueError(
                    "evaluation factory returned the live training adapter"
                )
            for index, seed in enumerate(evaluation.seeds):
                cursor = PolicyCursor(
                    actor,
                    adapter,
                    evaluation.environment.units,
                    stream_seed(seed, "evaluation-actions", 0),
                )
                completed_episode = False
                try:
                    cursor.reset(
                        adapter.reset(stream_seed(seed, "evaluation-reset", 0)),
                        f"evaluation-{index}",
                    )
                    for _ in range(evaluation.max_episode_steps):
                        step = cursor.step(
                            cursor.decide(
                                deterministic=evaluation.action_selection
                                == "deterministic"
                            )
                        )
                        if cursor.ended:
                            completed_episode = step.terminated
                            break
                finally:
                    episodes.append(
                        {
                            "seed": seed,
                            "return": cursor.episode_return,
                            "length": cursor.episode_length,
                            "completed": completed_episode,
                            "censored": not completed_episode,
                            "success": cursor.last_result.success
                            if completed_episode and cursor.last_result is not None
                            else None,
                        }
                    )
    except EvaluationTimeout:
        timed_out = True
    finally:
        try:
            if adapter is not None:
                adapter.close()
        finally:
            for module, training in training_modes.items():
                module.training = training
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
        "unattempted_count": len(evaluation.seeds) - len(episodes),
        "timed_out": timed_out,
        "episodes": episodes,
        "seconds": time.monotonic() - started,
        "action_selection": evaluation.action_selection,
    }
    if successes and all(success is not None for success in successes):
        result["success_rate"] = statistics.fmean(successes)
    if evaluation.monitor.metric == "score_mean":
        if timed_out or len(episodes) != len(evaluation.seeds):
            raise ValueError("GRPO evaluation requires every finite-horizon score")
        result["score_mean"] = statistics.fmean(
            episode["return"] for episode in episodes
        )
    result["selection_metric"] = evaluation.monitor.metric
    result["selection_value"] = result.get(evaluation.monitor.metric)
    result["selection_available"] = result["selection_value"] is not None
    return result
