"""Shared trajectory assembly and adapter cleanup for local collectors."""

import warnings

from sequifier.trajectory.schema import Arrival, Trajectory


class TrajectoryRecorder:
    def __init__(self, episode_id, max_episode_steps):
        self.episode_id, self.max_episode_steps = episode_id, max_episode_steps
        self.arrivals = []

    def append(self, *, terminated, truncated, **values):
        step = len(self.arrivals)
        transition = Arrival(
            terminated=terminated,
            truncated=truncated
            or (
                self.max_episode_steps is not None
                and step + 1 >= self.max_episode_steps
            ),
            **values,
        )
        self.arrivals.append(transition)
        return transition

    def finish(self, **values):
        return Trajectory(episode_id=self.episode_id, arrivals=self.arrivals, **values)


def close_adapter(adapter, failed):
    try:
        adapter.close()
    except Exception as exc:
        if not failed:
            raise
        warnings.warn(f"Adapter close also failed: {exc}", RuntimeWarning)
