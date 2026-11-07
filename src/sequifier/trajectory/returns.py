"""Exact recorded-reward returns and matching online return-budget arithmetic."""

from collections.abc import Sequence

from sequifier.config.policy_config import ReturnDefinition
from sequifier.rollout.history import numeric_scalar
from sequifier.trajectory.schema import Trajectory


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


def labeled_episode_length(
    trajectory: Trajectory, definition: ReturnDefinition
) -> int | None:
    """Exclude unknown tails; an explicit horizon labels only a fully observed prefix."""
    length = len(trajectory.transitions)
    horizon = definition.finite_horizon
    if horizon is not None and length >= horizon:
        return horizon
    if trajectory.transitions[-1].terminated:
        return length
    # A time limit is not a known terminal return without a finite-horizon objective.
    return None
