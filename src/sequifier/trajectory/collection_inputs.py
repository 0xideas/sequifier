"""Read historical collector archives using only the facts each archive recorded."""

from dataclasses import dataclass

import polars as pl

from sequifier.io.sequence import ordered_sequence_rows
from sequifier.trajectory.archives import read_collection_archive
from sequifier.trajectory.sequence_output import (
    action_record,
    decision_row,
    sequence_coordinate,
)
from sequifier.trajectory.tables import table_episode


@dataclass
class ArchiveEpisode:
    rows: list
    policy: object
    identity: str

    def load(self):
        frame = ordered_sequence_rows(
            pl.from_dicts(self.rows, infer_schema_length=None)
        )
        return table_episode(
            frame, self.policy, sequence_id=sequence_coordinate(self.identity)
        )


def collection_episodes(source, policy):
    from sequifier.config.policy_config import ResolvedPolicyBinding

    episodes = {}
    for path in source.paths:
        archive = read_collection_archive(path)
        namespace = source.declaration.episode_namespace
        if recorded := archive.get("sequence_namespace"):
            namespace += recorded + "/"
        if archive["format"] == "sequifier.ppo-collection":
            recording = ResolvedPolicyBinding.model_validate(archive["policy"])
            for step in archive["decisions"]:
                frame = archive["frames"][step["frame"]]
                if (frame["episode_id"], frame["step_id"]) != (
                    step["episode_id"],
                    step["step_id"],
                ):
                    raise ValueError(
                        "PPO archive decision disagrees with its recorded frame"
                    )
                identity = namespace + frame["episode_id"]
                successor = (
                    archive["frames"][step["successor"]]["observation"]
                    if step["successor"] is not None
                    else None
                )
                row = decision_row(
                    identity,
                    frame["step_id"],
                    frame["observation"],
                    frame["static_inputs"],
                    action=action_record(step["action"], recording),
                    previous=action_record(frame["previous_action"], recording),
                    reward=step["raw_reward"],
                    successor=successor,
                    terminated=step["terminated"],
                    truncated=step["truncated"],
                )
                episodes.setdefault(identity, []).append(row)
        else:
            for trajectory in archive["trajectories"]:
                frames = trajectory["frames"]
                identity = namespace + trajectory["episode_id"]
                rows = episodes.setdefault(identity, [])
                for i, (frame, encoded) in enumerate(
                    zip(frames, trajectory["actions"], strict=True)
                ):
                    last = i + 1 == len(frames)
                    row = decision_row(
                        identity,
                        frame["step_id"],
                        frame["observation"],
                        frame["static_inputs"],
                        action=action_record(encoded, policy),
                        previous=action_record(frame["previous_action"], policy),
                        reward=None,
                        successor=None if last else frames[i + 1]["observation"],
                        terminated=last and trajectory["cutoff_reason"] == "terminated",
                        truncated=last and trajectory["cutoff_reason"] == "truncated",
                        group=f"{namespace}group-{trajectory['group_index']}",
                    )
                    # Historical GRPO archives store scaled rewards and omit the final
                    # successor. They can supply imitation examples independently of IQL.
                    row.pop("bootstrap_allowed")
                    rows.append(row)
    return [
        ArchiveEpisode(rows, policy, identity) for identity, rows in episodes.items()
    ]
