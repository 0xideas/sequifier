"""Publish committed PPO and GRPO decisions independently of optimizer archives."""

from pathlib import Path
from uuid import uuid4

import polars as pl

from sequifier.fingerprints import file_checksum, fingerprint
from sequifier.io.atomic import write_json_atomic
from sequifier.io.sequence import register_sequence_name
from sequifier.trajectory.sequence_output import (
    action_record,
    decision_row,
    manifest_values,
    write_parts,
)


class CollectedSequences:
    def __init__(self, context, state):
        self.policy = context.actor.config
        self.output = context.phase.collection.output
        self.phase = context.phase.name
        self.algorithm = context.phase.algorithm
        self.parts = list(state.get("sequence_parts", ()))
        self.names_by_id = {}
        self.namespace = state.get("sequence_namespace") or (
            f"{context.metrics.run_id}/{context.index}"
            if context.metrics
            else uuid4().hex
        )
        if self.output:
            for part in self.parts:
                path = Path(self.output.manifest_path).parent / part["path"]
                if file_checksum(path) != part["sha256"]:
                    raise ValueError(f"Committed sequence shard changed: {path}")
                # Old checkpoints may own immutable shards from before the column rename.
                label = (
                    "sequenceName"
                    if "sequenceName" in pl.read_parquet_schema(path)
                    else "episodeId"
                )
                names = pl.read_parquet(path, columns=["sequenceId", label])
                for seq, name in names.unique().iter_rows():
                    register_sequence_name(self.names_by_id, seq, name)

    def checkpoint(self):
        return (
            {"sequence_namespace": self.namespace, "sequence_parts": self.parts}
            if self.output
            else {}
        )

    def row(self, frame, action, reward, successor, terminated, truncated, group):
        record = action_record(action, self.policy)
        assert record is not None
        row = decision_row(
            f"{self.namespace}/{frame.episode_id}",
            frame.step_id,
            dict(frame.observation),
            dict(frame.static_inputs),
            action=record,
            previous=action_record(frame.previous_action, self.policy),
            predictions={key: record[key] for key in self.policy.actions},
            reward=reward,
            successor=successor,
            terminated=terminated,
            truncated=truncated,
            group=group,
        )
        register_sequence_name(self.names_by_id, row["sequenceId"], row["sequenceName"])
        return row

    def ppo(self, buffer):
        if self.output:
            rows = (
                self.row(
                    buffer.frames[step.frame],
                    step.action,
                    step.raw_reward,
                    dict(buffer.frames[step.successor].observation)
                    if step.successor is not None
                    else None,
                    step.terminated,
                    step.truncated,
                    None,
                )
                for step in buffer.decisions
            )
            self.parts.extend(write_parts(self.output.manifest_path, rows))

    def grpo(self, buffer):
        if self.output:

            def rows():
                for trajectory in buffer.trajectories:
                    for i, frame in enumerate(trajectory.frames):
                        last = i + 1 == len(trajectory.actions)
                        successor = (
                            trajectory.final_observation
                            if last
                            else dict(trajectory.frames[i + 1].observation)
                        )
                        yield self.row(
                            frame,
                            trajectory.actions[i],
                            trajectory.raw_rewards[i],
                            successor,
                            last and trajectory.terminated,
                            last and trajectory.truncated,
                            f"{self.namespace}/group-{trajectory.group_index}",
                        )

            self.parts.extend(write_parts(self.output.manifest_path, rows()))

    def publish(self):
        if not self.output:
            return {}
        manifest = manifest_values(
            self.parts,
            self.policy.observation_columns,
            {
                "task": "policy",
                "phase": self.phase,
                "algorithm": self.algorithm,
                "policy_binding": self.policy.task_binding.model_dump(mode="json"),
            },
            static=self.policy.static_columns,
        )
        path = Path(self.output.manifest_path)
        pinned = {
            **manifest,
            "shards": [
                {**part, "path": Path(part["path"]).name} for part in self.parts
            ],
        }
        snapshot = path.with_name(path.name + ".parts") / (
            fingerprint(pinned) + ".manifest.json"
        )
        write_json_atomic(snapshot, pinned)
        write_json_atomic(path, manifest)
        return {
            "sequence_manifest": self.output.manifest_path,
            "committed_sequence_manifest": str(snapshot),
            "committed_sequence_manifest_sha256": file_checksum(snapshot),
            "sequence_manifest_sha256": file_checksum(self.output.manifest_path),
        }
