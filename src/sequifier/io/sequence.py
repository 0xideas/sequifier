"""Columnar demonstrations with optional transition capabilities.

These records describe data, independently of the simulator, collection context
length, prediction bindings, preprocessing or the algorithm that will consume it.
"""

from dataclasses import dataclass
from functools import cached_property

import polars as pl
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    TypeAdapter,
    model_validator,
)

from sequifier.config.column_names import NEXT_ACTION_OFFSET
from sequifier.io.values import DomainValue, domain_scalar

# Physical fields are kept separate from model features at the same coordinate.
DOMAIN_OBSERVATIONS = "__sequifier_domain_observations"
DOMAIN_STATIC = "__sequifier_domain_static"


def with_domain_record(frame, columns, name, mask_column=None):
    if name in frame.columns:
        raise ValueError(f"{name} is reserved for rollout seed values")
    if mask_column:
        from sequifier.io.feature_fitting import mask_expression

        if mask_column in columns:
            raise ValueError("The mask cannot also be an observation feature")
        frame = frame.with_columns(
            mask_expression(
                frame[mask_column], frame.schema[mask_column], mask_column
            ).alias(mask_column)
        )
    return frame.with_columns(pl.struct(columns).alias(name)) if columns else frame


def source_fields(rows, record, columns):
    """Select physical fields without changing the model feature columns."""
    if not columns:
        return []
    dtype = rows.schema.get(record)
    if not isinstance(dtype, pl.Struct) or set(columns) - set(dtype.to_schema()):
        raise ValueError(f"Prepared data lacks source fields in {record}: {columns}")
    return [pl.col(record).struct.field(name).alias(name) for name in columns]


def episode_identity(sequence_id, label=None):
    """Use a recorded domain label or the sequence coordinate as its external ID."""
    identity = str(sequence_id if label is None else domain_scalar(label, "episode ID"))
    if not identity:
        raise ValueError("episode ID must be nonempty")
    return identity


def register_sequence_name(names, sequence_id, name):
    """Check that one integer coordinate denotes one sequence across sources."""
    previous = names.setdefault(sequence_id, name)
    if previous != name:
        raise ValueError(
            f"sequenceId {sequence_id} refers to both {previous!r} and {name!r}"
        )


def validate_sequence_ids(frame, column="sequenceId"):
    ids = frame[column]
    if not ids.dtype.is_integer() or ids.null_count():
        raise ValueError(f"{column} must contain non-null integers")
    # Match the signed tensor coordinates used by prepared and supervised data.
    ids.cast(pl.Int64, strict=True)


def ordered_sequence_rows(frame, position_column="itemPosition"):
    """Share explicit temporal ordering and gap validation across table consumers."""
    if position_column not in frame.columns:
        raise ValueError(f"Missing coordinate {position_column}")
    positions = frame[position_column]
    if not positions.dtype.is_integer() or positions.null_count():
        raise ValueError(f"{position_column} must contain non-null integers")
    if not positions.is_sorted():
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

    def at(self, index: int) -> dict:
        """Read a transition without inventing absent capabilities."""
        return {
            name: values[index] if values is not None else None for name, values in self
        }


class ContextObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    observation: dict[str, JsonValue]
    masked: StrictBool = False
    previous_action: dict[str, DomainValue] | None = None
    reward: float | None = None
    desired_return: float | None = None

    def return_before(self, following, gamma):
        if self.reward is not None:
            return self.reward + gamma * following
        if self.desired_return is not None:
            return self.desired_return
        raise ValueError(
            "Return-conditioned history requires rewards or recorded budgets"
        )


@dataclass(frozen=True, slots=True)
class EpisodeDecision:
    decision_position: int
    target_position: int
    action: dict[str, DomainValue]
    reward: float | None
    terminated: bool | None
    truncated: bool | None


class SequenceEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    sequence_id: StrictInt | None = Field(default=None, ge=-(2**63), lt=2**63)
    episode_id: str = Field(min_length=1)
    context: tuple[ContextObservation, ...] = ()
    first_position: StrictInt = Field(default=0, ge=0)
    observations: dict[str, tuple[JsonValue, ...]] = Field(min_length=1)
    observation_masks: tuple[StrictBool, ...] = ()
    incoming_actions: tuple[dict[str, DomainValue] | None, ...]
    static_inputs: dict[str, JsonValue] = Field(default_factory=dict)
    group_id: str | None = None
    collection: tuple[dict[str, JsonValue] | None, ...] | None = None
    transitions: TransitionColumns | None = None

    @model_validator(mode="after")
    def validate_sequence(self):
        length = len(self.actions)
        rows = length + 1
        if "observation_masks" not in self.model_fields_set:
            object.__setattr__(self, "observation_masks", (False,) * rows)
        if (
            not length
            or len(self.observation_masks) != rows
            or any(
                len(values) != rows or any(value is None for value in values)
                for values in self.observations.values()
            )
        ):
            raise ValueError("N actions require N + 1 observations and masks")
        if set(self.observations) & set(self.static_inputs):
            raise ValueError("static and temporal features must be disjoint")
        if (self.first_position and self.previous_action is None) or any(
            action is None for action in self.incoming_actions[1:]
        ):
            raise ValueError("Only a reset row may omit its incoming action")
        if self.collection is not None and len(self.collection) != length:
            raise ValueError("Collection diagnostics must align with arrivals")
        transitions = self.transitions
        if transitions is None:
            return self
        for field, values in transitions:
            if values is not None and len(values) != length:
                raise ValueError(f"transition {field} must match sequence length")
        if any(
            any(flags[:-1])
            for flags in (transitions.terminated, transitions.truncated)
            if flags
        ):
            raise ValueError("A sequence cannot continue after an episode boundary")
        return self

    @cached_property
    def actions(self):
        return self.incoming_actions[1:]

    @property
    def previous_action(self):
        return self.incoming_actions[0]

    @property
    def starts_episode(self):
        return self.first_position == 0 and self.previous_action is None

    @property
    def positions(self):
        return range(self.first_position, self.first_position + len(self.actions) + 1)

    def decision(self, step: int) -> EpisodeDecision:
        """Read the action and available outcomes of one episode-local decision."""
        if not 0 <= step < len(self.actions):
            raise IndexError(step)
        position = self.first_position + step
        action = self.incoming_actions[step + NEXT_ACTION_OFFSET]
        assert action is not None
        outcome = self.transitions.at(step) if self.transitions else {}
        return EpisodeDecision(
            decision_position=position,
            target_position=position + NEXT_ACTION_OFFSET,
            action=action,
            reward=outcome.get("rewards"),
            terminated=outcome.get("terminated"),
            truncated=outcome.get("truncated"),
        )

    def observation(self, index: int) -> dict:
        return {column: values[index] for column, values in self.observations.items()}

    def return_length(self, finite_horizon: int | None) -> int | None:
        if self.transitions is None or self.transitions.rewards is None:
            return None
        length = len(self.actions)
        if finite_horizon is not None and (
            self.starts_episode or self.first_position > 0
        ):
            remaining = finite_horizon - self.first_position
            if remaining <= 0:
                return None
            if length >= remaining:
                return remaining
        if self.transitions.terminated is not None and self.transitions.terminated[-1]:
            return length
        return None


# Sequence metadata carries the history supplied when collection starts.
CONTEXT_RECORDS = TypeAdapter(tuple[ContextObservation, ...])


def seed_metadata(context):
    return {
        "context": CONTEXT_RECORDS.dump_json(context).decode() if context else None,
    }
