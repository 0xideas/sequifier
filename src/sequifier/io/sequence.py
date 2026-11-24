"""Columnar demonstrations with optional transition capabilities.

These records describe data, independently of the simulator, collection context
length, prediction bindings, preprocessing or the algorithm that will consume it.
"""

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

from sequifier.io.values import DomainValue, domain_scalar

# Physical fields are kept separate from model features at the same coordinate.
DOMAIN_OBSERVATIONS = "__sequifier_domain_observations"
DOMAIN_STATIC = "__sequifier_domain_static"
OBSERVATION_MASK = "__observation_mask"
DOMAIN_SUCCESSORS = "__sequifier_domain_successors"
DOMAIN_CONTEXT = "__sequifier_domain_context"


def with_domain_record(frame, columns, name, mask_column=None):
    if name in frame.columns:
        raise ValueError(f"{name} is reserved for rollout seed values")
    if OBSERVATION_MASK in columns:
        raise ValueError(f"{OBSERVATION_MASK} is reserved for observation masking")
    if mask_column:
        from sequifier.io.feature_fitting import mask_expression

        if OBSERVATION_MASK in frame.columns and mask_column != OBSERVATION_MASK:
            raise ValueError(f"{OBSERVATION_MASK} is reserved for observation masking")
        frame = frame.with_columns(
            mask_expression(
                frame[mask_column], frame.schema[mask_column], mask_column
            ).alias(OBSERVATION_MASK)
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
    successors: tuple[dict[str, JsonValue] | None, ...] | None = None
    bootstrap_allowed: tuple[StrictBool, ...] | None = None

    def at(self, index: int) -> dict:
        """Read a transition without inventing absent capabilities."""
        return {
            name: values[index] if values is not None else None for name, values in self
        }


class SequenceEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    sequence_id: StrictInt | None = Field(default=None, ge=-(2**63), lt=2**63)
    episode_id: str = Field(min_length=1)
    item_positions: tuple[StrictInt, ...] = Field(min_length=1)
    observations: dict[str, tuple[JsonValue, ...]] = Field(min_length=1)
    observation_masks: tuple[StrictBool, ...] = ()
    actions: tuple[DomainValue | dict[str, DomainValue], ...]
    static_inputs: dict[str, JsonValue] = Field(default_factory=dict)
    starts_episode: StrictBool = True
    previous_action: DomainValue | dict[str, DomainValue] | None = None
    group_id: str | None = None
    transitions: TransitionColumns | None = None

    @model_validator(mode="after")
    def validate_sequence(self):
        length = len(self.item_positions)
        if "observation_masks" not in self.model_fields_set:
            object.__setattr__(self, "observation_masks", (False,) * length)
        if len(self.observation_masks) != length:
            raise ValueError("Observation masks must match decision count")
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
        for field, values in transitions:
            if values is not None and len(values) != length:
                raise ValueError(f"transition {field} must match sequence length")
        for index in range(length):
            step = transitions.at(index)
            terminated, truncated = step["terminated"], step["truncated"]
            if (terminated or truncated) and index != length - 1:
                raise ValueError(
                    "a sequence cannot contain a decision after an episode boundary"
                )
            successor = step["successors"]
            if successor is not None:
                if successor.keys() != self.observations.keys():
                    raise ValueError(
                        "successor observation fields must match observations"
                    )
                if index + 1 < length and successor != self.observation(index + 1):
                    raise ValueError("explicit successor breaks sequence continuity")
            bootstrap = step["bootstrap_allowed"]
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


class PreparedEpisode(SequenceEpisode):
    """Split-owned decisions plus preceding observations used only as model inputs."""

    context: tuple[ContextObservation, ...]


# An optional first-row column carries pre-decision history across table exports.
CONTEXT_COLUMN = "__policy_seed_context"
CONTEXT_RECORDS = TypeAdapter(tuple[ContextObservation, ...])
