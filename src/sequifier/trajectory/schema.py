"""Inspectable forecast and policy collection records."""

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    JsonValue,
    StrictInt,
    TypeAdapter,
    model_validator,
)

from sequifier.config.column_names import INCOMING_ACTION_PRESENT
from sequifier.config.rollout_config import TrajectoryInputBinding
from sequifier.fingerprints import fingerprint
from sequifier.io.values import DomainValue


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SeedHistory(Record):
    observations: dict[str, list[JsonValue]]
    attention_valid_mask: list[bool]
    observation_mask: list[bool] = Field(default_factory=list)
    static_inputs: dict[str, JsonValue]
    incoming_actions: list[dict[str, DomainValue] | None] | None = None

    @classmethod
    def from_rows(cls, rows, columns, width, static, masks, incoming_actions=None):
        padding = width - len(rows)
        if padding < 0:
            raise ValueError("Seed exceeds the model context length")
        return cls(
            observations={
                name: [0.0] * padding + [row[name] for row in rows] for name in columns
            },
            attention_valid_mask=[False] * padding + [True] * len(rows),
            observation_mask=[False] * padding + list(masks),
            static_inputs=static,
            incoming_actions=([None] * padding + incoming_actions)
            if incoming_actions is not None
            else None,
        )

    @property
    def valid_actions(self):
        count = sum(self.attention_valid_mask)
        return (
            self.incoming_actions[-count:] if self.incoming_actions else [None] * count
        )

    @model_validator(mode="after")
    def validate_window(self):
        mask = self.attention_valid_mask
        if (
            not mask
            or not mask[-1]
            or any(mask[i] and not mask[i + 1] for i in range(len(mask) - 1))
        ):
            raise ValueError("seed mask must be left padding followed by valid history")
        if self.incoming_actions is not None and len(self.incoming_actions) != len(
            mask
        ):
            raise ValueError("Incoming actions must align with seed observations")
        if "observation_mask" not in self.model_fields_set:
            self.observation_mask = [False] * len(mask)
        if len(self.observation_mask) != len(mask) or any(
            hidden and not valid
            for hidden, valid in zip(self.observation_mask, mask, strict=True)
        ):
            raise ValueError("Observation masks must align with valid seed positions")
        if not self.observations or any(
            len(values) != len(mask) for values in self.observations.values()
        ):
            raise ValueError(
                "seed feature histories must be nonempty and match the attention mask length"
            )
        if set(self.observations) & set(self.static_inputs):
            raise ValueError("static inputs must be separate from changing inputs")
        return self


class Arrival(Record):
    observation: dict[str, JsonValue]
    action: dict[str, DomainValue]
    reward: FiniteFloat
    terminated: bool
    truncated: bool
    action_latents: dict[str, FiniteFloat] = Field(default_factory=dict)
    action_components: dict[str, StrictInt] = Field(default_factory=dict)
    diagnostics: dict[str, JsonValue] = Field(default_factory=dict)


class Trajectory(Record):
    episode_id: str = Field(min_length=1)
    seed: SeedHistory
    bindings: list[TrajectoryInputBinding]
    prediction_columns: list[str]
    arrivals: list[Arrival] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_arrivals(self):
        for index, row in enumerate(self.arrivals):
            if row.observation.keys() != self.seed.observations.keys() or set(
                row.action
            ) != set(self.prediction_columns):
                raise ValueError(
                    f"Arrival {index + 1} fields disagree with the trajectory"
                )
            if (row.terminated or row.truncated) and index + 1 != len(self.arrivals):
                raise ValueError("An episode boundary must be the last arrival")
        return self

    @property
    def observation_columns(self):
        return [
            name for name in self.seed.observations if name != INCOMING_ACTION_PRESENT
        ]

    @property
    def collection_cutoff(self):
        last = self.arrivals[-1]
        return not (last.terminated or last.truncated)

    @property
    def seed_group(self):
        origin = self.metadata.get("seed_origin")
        keys = ("dataset", "sequenceId", "itemPosition")
        if isinstance(origin, dict) and all(key in origin for key in keys):
            return fingerprint({key: origin[key] for key in keys})
        return None


def validate_seed_budgets(spec, budgets, count, desired_return):
    budgets = TypeAdapter(list[FiniteFloat]).validate_python(budgets, strict=True)
    if len(budgets) != (count if spec.return_conditioned else 0) or (
        budgets and budgets[-1] != desired_return
    ):
        raise ValueError("Seed budgets must align with the seed history")
