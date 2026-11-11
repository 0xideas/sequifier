"""Shared trajectory assembly and adapter cleanup for local collectors."""

import math
import warnings

from sequifier.trajectory.schema import Trajectory, Transition


class TrajectoryRecorder:
    def __init__(self, episode_id, max_episode_steps):
        self.episode_id, self.max_episode_steps = episode_id, max_episode_steps
        self.transitions = []

    def append(self, *, terminated, truncated, **values):
        step = len(self.transitions)
        transition = Transition(
            episode_id=self.episode_id,
            step_id=step,
            terminated=terminated,
            truncated=truncated
            or (
                self.max_episode_steps is not None
                and step + 1 >= self.max_episode_steps
            ),
            bootstrap_allowed=not terminated,
            **values,
        )
        self.transitions.append(transition)
        return transition

    def finish(self, **values):
        last = self.transitions[-1]
        return Trajectory(
            episode_id=self.episode_id,
            transitions=self.transitions,
            collection_cutoff=not (last.terminated or last.truncated),
            total_return=math.fsum(step.reward for step in self.transitions),
            **values,
        )


def close_adapter(adapter, failed):
    try:
        adapter.close()
    except Exception as exc:
        if not failed:
            raise
        warnings.warn(f"Adapter close also failed: {exc}", RuntimeWarning)
