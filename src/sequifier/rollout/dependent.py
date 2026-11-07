"""Small local dependent-feature adapter, also usable as an integration example."""

from sequifier.rollout.contracts import ResetRequest, StepRequest, StepResult


class LinearDependentAdapter:
    """Supply a scaled prediction and use that prediction as the reward."""

    protocol_version = 1

    def __init__(
        self, prediction="x", external_input="d", multiplier=2.0, terminate_after=3
    ):
        if (
            not isinstance(terminate_after, int)
            or isinstance(terminate_after, bool)
            or terminate_after < 1
        ):
            raise ValueError("terminate_after must be a positive integer")
        self.prediction = prediction
        self.external_input = external_input
        self.multiplier = multiplier
        self.terminate_after = terminate_after
        self.closed = False
        self.steps = 0
        self.episode_id = None

    def reset(self, request: ResetRequest) -> None:
        if self.closed:
            raise ValueError("adapter is closed")
        self.steps = 0
        self.episode_id = request.episode_id

    def step(self, request: StepRequest) -> StepResult:
        if (
            self.closed
            or self.episode_id != request.episode_id
            or self.steps >= self.terminate_after
        ):
            raise ValueError("adapter must be reset before stepping an active episode")
        self.steps += 1
        prediction = request.predictions[self.prediction]
        return StepResult(
            {self.external_input: self.multiplier * prediction},
            reward=prediction,
            terminated=self.steps == self.terminate_after,
        )

    def close(self) -> None:
        self.closed = True
