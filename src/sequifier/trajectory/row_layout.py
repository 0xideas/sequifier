"""Physical layout of a contiguous trajectory table."""

from sequifier.config.column_names import (
    INCOMING_ACTION_PRESENT,
    TRANSITION_COLUMNS,
    validate_column_names,
)
from sequifier.io.values import portable_value

COORDINATES = ("sequenceId", "itemPosition", "sequenceName", "groupId")
ACTION_METADATA = "actionMetadata"


class TrajectoryRowLayout:
    def __init__(
        self, observations, static, actions, *, mask_column=None, aliases=None
    ):
        features = {*observations, *static}
        reserved = {
            *COORDINATES,
            *TRANSITION_COLUMNS,
            ACTION_METADATA,
            "__collection",
            INCOMING_ACTION_PRESENT,
        }
        occupied = features | reserved
        if mask_column is None:
            mask_column = "observationMasked"
            while mask_column in occupied or mask_column in actions:
                mask_column += "_mask"
        self.mask_column = mask_column
        validate_column_names([self.mask_column], "mask_column")
        if self.mask_column in occupied:
            raise ValueError(
                f"Mask column {self.mask_column!r} overlaps trajectory fields"
            )
        if features & reserved:
            raise ValueError("Rollout features overlap trajectory metadata")
        occupied.add(self.mask_column)
        aliases = aliases or {}
        if set(aliases) - set(actions):
            raise ValueError("Sequence output aliases must name declared actions")
        self.actions = {}
        for name in actions:
            physical = aliases.get(name, name)
            if name not in aliases and physical in occupied:
                physical = name + "_action"
                while (
                    physical in occupied
                    or physical in actions
                    or physical in aliases.values()
                ):
                    physical += "_action"
            validate_column_names([physical], "trajectory action columns")
            if physical in occupied:
                raise ValueError(
                    f"Action column {physical!r} overlaps trajectory fields"
                )
            self.actions[name] = physical
            occupied.add(physical)

    def row(
        self,
        coordinates,
        observation,
        static,
        *,
        action,
        reward=None,
        terminated=None,
        truncated=None,
        masked=False,
    ):
        metadata = {
            key: value
            for key, value in (action or {}).items()
            if key not in self.actions
        }
        values = {
            **coordinates,
            **observation,
            **static,
            self.mask_column: masked,
            **{
                physical: action[logical] if action is not None else None
                for logical, physical in self.actions.items()
            },
            ACTION_METADATA: metadata or None,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
        return {name: portable_value(value) for name, value in values.items()}

    def manifest_fields(self):
        return {"action_columns": self.actions, "mask_column": self.mask_column}
