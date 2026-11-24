"""Episode readers for resolved tables, prepared histories, and rollout logs."""

from dataclasses import dataclass
from pathlib import Path

from sequifier.config.policy_config import ResolvedPolicyBinding
from sequifier.config.policy_inputs import PolicyEpisodeSpec
from sequifier.config.policy_train_config import SequenceSourceConfig
from sequifier.io.action_record import action_record_layout
from sequifier.io.actions import decode_action
from sequifier.io.sequence import (
    ContextObservation,
    PreparedEpisode,
    SequenceEpisode,
    TransitionColumns,
)
from sequifier.trajectory.schema import Trajectory, validate_trajectory
from sequifier.trajectory.sources import ResolvedSequenceSource, resolve_source
from sequifier.trajectory.store import load_trajectory
from sequifier.trajectory.tables import table_episodes


def sequence_from_trajectory(
    trajectory: Trajectory, policy: PolicyEpisodeSpec, action_semantics="policy"
) -> SequenceEpisode:
    from sequifier.trajectory.sequence_output import sequence_coordinate

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
    execution = episode.metadata.get("policy_execution", {})
    recording_policy = (
        ResolvedPolicyBinding.model_validate(execution["config"])
        if execution.get("seed_previous_action_ids") is not None
        else policy
    )
    seed_actions = execution.get("seed_previous_action_ids")
    if seed_actions is None:
        count = (
            1
            if policy.previous_action_inputs or policy.return_conditioned
            else sum(episode.seed.attention_valid_mask)
        )
        seed_actions = [None] * count
    seed_returns = execution.get("seed_returns") or [None] * len(seed_actions)
    context = tuple(
        ContextObservation(
            observation={
                name: values[i - len(seed_actions)]
                for name, values in episode.seed.observations.items()
            },
            previous_action=decode_action(action, recording_policy)
            if action is not None
            else None,
            desired_return=seed_returns[i],
        )
        for i, action in enumerate(seed_actions[:-1])
    )
    previous = seed_actions[-1]
    result = SequenceEpisode(
        sequence_id=sequence_coordinate(episode.episode_id),
        episode_id=episode.episode_id,
        group_id=episode.seed_group,
        item_positions=tuple(step.step_id for step in steps),
        starts_episode=previous is None,
        previous_action=decode_action(previous, recording_policy)
        if previous is not None
        else None,
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

    return (
        PreparedEpisode(**result.model_dump(), context=context) if context else result
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
        from sequifier.trajectory.sequence_output import sequence_coordinate

        identity = self.namespace + episode.episode_id
        return episode.model_copy(
            update={
                "episode_id": identity,
                "sequence_id": sequence_coordinate(identity),
            }
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
