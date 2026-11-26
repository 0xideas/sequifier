"""Publish PPO and GRPO transitions and collection diagnostics in one decision table."""

import json
from pathlib import Path
from uuid import uuid4

import polars as pl
import pyarrow as pa

from sequifier.fingerprints import file_checksum
from sequifier.io.sequence import register_sequence_name
from sequifier.training.distributed_strategy import on_writer
from sequifier.trajectory.sequence_output import (
    action_record,
    decision_row,
    publish_sequences,
    write_parts,
)


def _collection_type(algorithm, output):
    fields = {"policy_version": pa.int64(), "scaled_reward": pa.float64()}
    if algorithm == "ppo":
        fields["collection_end"] = pa.bool_()
        scores = (
            (
                "old_log_probability",
                "old_value",
                "entropy",
                "successor_value",
                "advantage",
                "value_target",
            )
            if output.include_scores
            else ()
        )
    else:
        fields.update(
            reference_identity=pa.string(),
            group_index=pa.int64(),
            sample_index=pa.int64(),
            start_token_json=pa.string(),
            cutoff_reason=pa.string(),
        )
        scores = (
            "old_log_probability",
            "reference_log_probability",
            "score",
            "advantage",
        )
    fields.update(dict.fromkeys(scores, pa.float64()))
    return pa.struct(fields)


class CollectedSequences:
    def __init__(self, context, state):
        self.strategy = context.strategy
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
            self.namespace = self.strategy.broadcast(self.namespace)
            if "__collection" in (
                *self.policy.observation_columns,
                *self.policy.static_columns,
            ):
                raise ValueError("__collection is reserved for rollout diagnostics")
            on_writer(self.strategy, self._restore_parts)

    def _restore_parts(self):
        for part in self.parts:
            path = Path(self.output.manifest_path).parent / part["path"]
            if file_checksum(path) != part["sha256"]:
                raise ValueError(f"Committed sequence shard changed: {path}")
            names = pl.read_parquet(path, columns=["sequenceId", "sequenceName"])
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
        return decision_row(
            f"{self.namespace}/{frame.episode_id}",
            frame.step_id,
            dict(frame.observation),
            dict(frame.static_inputs),
            action=record,
            previous=action_record(frame.previous_action, self.policy),
            predictions={key: record[key] for key in self.policy.actions},
            masked=frame.masked,
            reward=reward,
            successor=successor,
            terminated=terminated,
            truncated=truncated,
            group=group,
        )

    def _write(self, rows):
        names = {}

        def checked_rows():
            for row in rows:
                register_sequence_name(names, row["sequenceId"], row["sequenceName"])
                yield row

        parts = list(
            write_parts(
                self.output.manifest_path,
                checked_rows(),
                policy=self.policy,
                observations=self.policy.observation_columns,
                collection_type=_collection_type(self.algorithm, self.output),
            )
        )

        return parts, names

    def record(self, buffer):
        if not self.output:
            return
        try:
            parts, names = [], {}
            if buffer is not None:
                write = self.ppo if self.algorithm == "ppo" else self.grpo
                parts, names = write(buffer)
            packet = parts, names, None
        except Exception as error:
            if self.strategy.world_size == 1:
                raise
            packet = [], {}, str(error)
        gathered = self.strategy.gather_for_writer(packet)

        def commit():
            new_parts = []
            new_names = {}
            for rank, (parts, names, error) in enumerate(gathered):
                if error is not None:
                    raise RuntimeError(f"Rollout writer rank {rank} failed: {error}")
                new_parts.extend(parts)
                for seq, name in names.items():
                    if seq in self.names_by_id:
                        register_sequence_name(self.names_by_id, seq, name)
                    else:
                        register_sequence_name(new_names, seq, name)
            self.parts.extend(new_parts)
            self.names_by_id.update(new_names)

        on_writer(self.strategy, commit)

    def ppo(self, buffer):
        def rows():
            for i, step in enumerate(buffer.decisions):
                row = self.row(
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
                details = {
                    "policy_version": buffer.policy_version,
                    "scaled_reward": step.reward,
                    "collection_end": i == len(buffer.decisions) - 1,
                }
                if self.output.include_scores:
                    details.update(
                        old_log_probability=step.old_log_probability,
                        old_value=step.old_value,
                        entropy=step.entropy,
                        successor_value=step.successor_value,
                        advantage=buffer.advantages[i],
                        value_target=buffer.targets[i],
                    )
                row["__collection"] = details
                yield row

        return self._write(rows())

    def grpo(self, buffer):
        def rows():
            for trajectory in buffer.trajectories:
                for i, frame in enumerate(trajectory.frames):
                    last = i + 1 == len(trajectory.actions)
                    successor = (
                        trajectory.final_observation
                        if last
                        else dict(trajectory.frames[i + 1].observation)
                    )
                    row = self.row(
                        frame,
                        trajectory.actions[i],
                        trajectory.raw_rewards[i],
                        successor,
                        last and trajectory.terminated,
                        last and trajectory.truncated,
                        f"{self.namespace}/group-{trajectory.group_index}",
                    )
                    row["__collection"] = {
                        "policy_version": trajectory.generation,
                        "reference_identity": buffer.reference_identity,
                        "group_index": trajectory.group_index,
                        "sample_index": trajectory.sample_index,
                        "start_token_json": json.dumps(
                            trajectory.start_token, allow_nan=False, sort_keys=True
                        ),
                        "scaled_reward": trajectory.rewards[i],
                        "old_log_probability": trajectory.old_log_probabilities[i],
                        "reference_log_probability": trajectory.reference_log_probabilities[
                            i
                        ],
                        "score": trajectory.score,
                        "advantage": trajectory.advantage,
                        "cutoff_reason": trajectory.cutoff_reason if last else None,
                    }
                    yield row

        return self._write(rows())

    def publish(self):
        if not self.output or self.strategy.rank != 0:
            return {}
        return publish_sequences(
            self.output.manifest_path,
            self.parts,
            self.policy.observation_columns,
            {
                "task": "policy",
                "phase": self.phase,
                "algorithm": self.algorithm,
                "policy_binding": self.policy.task_binding.model_dump(mode="json"),
            },
            static=self.policy.static_columns,
            pin=True,
        )
