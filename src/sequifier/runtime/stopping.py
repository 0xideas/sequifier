"""Serializable observation-based patience shared by native learning runners."""

import math
from dataclasses import dataclass


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
        improved = self.best is None or (
            value < self.best - config.min_delta
            if direction == "min"
            else value > self.best + config.min_delta
        )
        if improved:
            self.best = float(value)
            self.without_improvement = 0
        else:
            self.without_improvement += 1
        self.triggered = self.without_improvement >= config.patience
        return self.triggered
