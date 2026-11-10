"""Episode readers for resolved tables, prepared histories, and rollout logs."""

from dataclasses import dataclass
from pathlib import Path

from sequifier.config.policy_config import ResolvedPolicyBinding
from sequifier.config.policy_train_config import SequenceSourceConfig
from sequifier.io.actions import action_component_key, action_latent_key
from sequifier.io.sequence import SequenceEpisode, TransitionColumns
from sequifier.trajectory.schema import Trajectory, validate_trajectory
from sequifier.trajectory.sources import ResolvedSequenceSource, resolve_source
from sequifier.trajectory.store import load_trajectory
from sequifier.trajectory.tables import table_episodes


def sequence_from_trajectory(
    trajectory: Trajectory, policy: ResolvedPolicyBinding, action_semantics="policy"
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
    steps = episode.transitions
    return SequenceEpisode(
        episode_id=episode.episode_id,
        item_positions=tuple(step.step_id for step in steps),
        observations={
            column: tuple(step.observation[column] for step in steps)
            for column in episode.seed.observations
        },
        actions=tuple(
            {
                **step.predictions,
                **{
                    action_latent_key(name, policy): value
                    for name, value in step.action_latents.items()
                },
                **{
                    action_component_key(name, policy): value
                    for name, value in step.action_components.items()
                },
            }
            for step in steps
        ),
        static_inputs=episode.seed.static_inputs,
        transitions=TransitionColumns(
            **{
                field: tuple(getattr(step, attribute) for step in steps)
                for field, attribute in {
                    "rewards": "reward",
                    "terminated": "terminated",
                    "truncated": "truncated",
                    "successors": "successor_observation",
                    "bootstrap_allowed": "bootstrap_allowed",
                }.items()
            }
        ),
    )


@dataclass(frozen=True)
class TrajectoryReference:
    path: Path
    policy: ResolvedPolicyBinding
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
    source: SequenceSourceConfig | ResolvedSequenceSource, policy: ResolvedPolicyBinding
):
    resolved: ResolvedSequenceSource = (
        resolve_source(source) if isinstance(source, SequenceSourceConfig) else source
    )
    source_config = resolved.source
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
    return table_episodes(
        source_config, policy, resolved.paths, prepared=resolved.kind == "prepared"
    )
