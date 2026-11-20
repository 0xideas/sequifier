"""Episode readers for resolved tables, prepared histories, and rollout logs."""

from dataclasses import dataclass
from pathlib import Path

from sequifier.config.policy_inputs import PolicyEpisodeSpec
from sequifier.config.policy_train_config import SequenceSourceConfig
from sequifier.io.action_record import action_record_layout
from sequifier.io.sequence import SequenceEpisode, TransitionColumns
from sequifier.trajectory.schema import Trajectory, validate_trajectory
from sequifier.trajectory.sources import ResolvedSequenceSource, resolve_source
from sequifier.trajectory.store import load_trajectory
from sequifier.trajectory.tables import table_episodes


def sequence_from_trajectory(
    trajectory: Trajectory, policy: PolicyEpisodeSpec, action_semantics="policy"
) -> SequenceEpisode:
    episode = validate_trajectory(trajectory)
    if (
        episode.format
        not in {"sequifier.scalar-policy-trajectory", "sequifier.policy-trajectory"}
        and action_semantics != "executed_predictions"
    ):
        raise ValueError(
            "forecast sources require explicit action_semantics='executed_predictions'"
        )
    if episode.prediction_columns != policy.action_columns:
        raise ValueError("recorded actions do not match the policy action column")
    layout = action_record_layout(policy)
    steps = episode.transitions
    return SequenceEpisode(
        episode_id=episode.episode_id,
        item_positions=tuple(step.step_id for step in steps),
        observations={
            column: tuple(step.observation[column] for step in steps)
            for column in episode.seed.observations
        },
        actions=tuple(
            layout.pack(step.predictions, step.action_latents, step.action_components)
            for step in steps
        ),
        static_inputs=episode.seed.static_inputs,
        transitions=TransitionColumns(
            rewards=tuple(step.reward for step in steps),
            terminated=tuple(step.terminated for step in steps),
            truncated=tuple(step.truncated for step in steps),
            successors=tuple(step.successor_observation for step in steps),
            bootstrap_allowed=tuple(step.bootstrap_allowed for step in steps),
        ),
    )


@dataclass(frozen=True)
class TrajectoryReference:
    path: Path
    policy: PolicyEpisodeSpec
    action_semantics: str = "policy"
    namespace: str = ""

    def load(self) -> SequenceEpisode:
        episode = sequence_from_trajectory(
            load_trajectory(self.path), self.policy, self.action_semantics
        )
        return episode.model_copy(
            update={"episode_id": self.namespace + episode.episode_id}
        )


def source_episodes(
    source: SequenceSourceConfig | ResolvedSequenceSource, policy: PolicyEpisodeSpec
):
    resolved: ResolvedSequenceSource = (
        resolve_source(source) if isinstance(source, SequenceSourceConfig) else source
    )
    source_config = resolved.declaration
    if resolved.kind == "trajectory":
        return [
            TrajectoryReference(
                path,
                policy,
                source_config.action_semantics,
                source_config.episode_namespace,
            )
            for path in resolved.paths
        ]
    return table_episodes(resolved, policy)
