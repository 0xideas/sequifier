"""Exact recorded-reward returns and matching online return-budget arithmetic."""

from collections.abc import Sequence

from sequifier.rollout.history import numeric_scalar


def return_to_go(rewards: Sequence[float], gamma: float = 1.0) -> list[float]:
    gamma = numeric_scalar(gamma, "gamma")
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    result = [0.0] * len(rewards)
    remaining = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        remaining = numeric_scalar(
            numeric_scalar(rewards[index], "reward") + gamma * remaining, "return-to-go"
        )
        result[index] = remaining
    return result


def remaining_return(desired_return: float, reward: float, gamma: float = 1.0) -> float:
    """If G_t = r_t + gamma G_(t+1), then G_(t+1) = (G_t-r_t)/gamma."""
    gamma = numeric_scalar(gamma, "gamma")
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    return numeric_scalar(
        (
            numeric_scalar(desired_return, "desired_return")
            - numeric_scalar(reward, "reward")
        )
        / gamma,
        "successor desired_return",
    )
