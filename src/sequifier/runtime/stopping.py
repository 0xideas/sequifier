"""Serializable observation-based patience shared by native learning runners."""

import math
from dataclasses import dataclass

from sequifier.runtime.plan import metric_improved


@dataclass
class StoppingState:
    best: float | None = None
    without_improvement: int = 0
    last_observation: int | None = None
    triggered: bool = False

    def observe(self, config, value, direction, observation):
        if config is None or value is None or not math.isfinite(value):
            return self.triggered
        if self.last_observation is not None and observation <= self.last_observation:
            return self.triggered
        self.last_observation = observation
        if metric_improved(value, self.best, direction, config.min_delta):
            self.best = float(value)
            self.without_improvement = 0
        else:
            self.without_improvement += 1
        self.triggered = self.without_improvement >= config.patience
        return self.triggered
