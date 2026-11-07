"""Columnar demonstrations with optional transition capabilities.

These records describe data, independently of the simulator, collection context
length, prediction bindings, preprocessing or the algorithm that will consume it.
"""

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    model_validator,
)

from sequifier.io.values import DomainValue


def ordered_sequence_rows(frame, position_column="itemPosition"):
    """Share explicit temporal ordering and gap validation across table consumers."""
    if position_column not in frame.columns:
        return frame
    positions = frame[position_column]
    if not positions.dtype.is_integer() or positions.null_count():
        raise ValueError(f"{position_column} must contain non-null integers")
    frame = frame.sort(position_column)
    values = frame[position_column].to_list()
    if values and (
        values[0] < 0
        or any(right != left + 1 for left, right in zip(values, values[1:]))
    ):
        raise ValueError(
            f"{position_column} must be nonnegative, unique and contiguous within a sequence"
        )
    return frame


class TransitionColumns(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    rewards: tuple[float, ...] | None = None
    terminated: tuple[StrictBool, ...] | None = None
    truncated: tuple[StrictBool, ...] | None = None
    # None at a position means the true successor was not observed. Never a reset.
    successors: tuple[dict[str, DomainValue] | None, ...] | None = None
    bootstrap_allowed: tuple[StrictBool, ...] | None = None


class SequenceEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    episode_id: str = Field(min_length=1)
    item_positions: tuple[StrictInt, ...] = Field(min_length=1)
    observations: dict[str, tuple[DomainValue, ...]] = Field(min_length=1)
    actions: tuple[DomainValue, ...]
    static_inputs: dict[str, DomainValue] = Field(default_factory=dict)
    starts_episode: StrictBool = True
    previous_action: DomainValue | None = None
    group_id: str | None = None
    transitions: TransitionColumns | None = None

    @model_validator(mode="after")
    def validate_sequence(self):
        length = len(self.item_positions)
        if self.item_positions[0] < 0 or any(
            right != left + 1
            for left, right in zip(self.item_positions, self.item_positions[1:])
        ):
            raise ValueError(
                "itemPosition must be nonnegative, unique and contiguous within an episode"
            )
        if len(self.actions) != length or any(
            len(values) != length for values in self.observations.values()
        ):
            raise ValueError(
                "sequence feature/action lengths must match item positions"
            )
        if set(self.observations) & set(self.static_inputs):
            raise ValueError("static and temporal features must be disjoint")
        if self.starts_episode:
            if self.item_positions[0] != 0 or self.previous_action is not None:
                raise ValueError(
                    "episode starts require position zero and no preceding action"
                )
        elif self.previous_action is None:
            raise ValueError("fragments require an explicit preceding action")
        transitions = self.transitions
        if transitions is None:
            return self
        for field in type(transitions).model_fields:
            values = getattr(transitions, field)
            if values is not None and len(values) != length:
                raise ValueError(f"transition {field} must match sequence length")
        for index in range(length):
            terminated = (
                transitions.terminated is not None and transitions.terminated[index]
            )
            truncated = (
                transitions.truncated is not None and transitions.truncated[index]
            )
            if (terminated or truncated) and index != length - 1:
                raise ValueError(
                    "a sequence cannot contain a decision after an episode boundary"
                )
            successor = (
                transitions.successors[index]
                if transitions.successors is not None
                else None
            )
            if successor is not None:
                if successor.keys() != self.observations.keys():
                    raise ValueError(
                        "successor observation fields must match observations"
                    )
                if index + 1 < length and successor != self.observation(index + 1):
                    raise ValueError("explicit successor breaks sequence continuity")
            bootstrap = (
                transitions.bootstrap_allowed[index]
                if transitions.bootstrap_allowed is not None
                else False
            )
            if terminated and bootstrap:
                raise ValueError("task termination must disable bootstrap")
            if bootstrap and index + 1 == length and successor is None:
                raise ValueError(
                    "bootstrapped final decisions require an observed successor"
                )
        return self

    def observation(self, index: int) -> dict:
        return {column: values[index] for column, values in self.observations.items()}

    def return_length(self, finite_horizon: int | None) -> int | None:
        if self.transitions is None or self.transitions.rewards is None:
            return None
        length = len(self.actions)
        if finite_horizon is not None and (
            self.starts_episode or self.item_positions[0] > 0
        ):
            remaining = finite_horizon - self.item_positions[0]
            if remaining <= 0:
                return None
            if length >= remaining:
                return remaining
        if self.transitions.terminated is not None and self.transitions.terminated[-1]:
            return length
        return None
