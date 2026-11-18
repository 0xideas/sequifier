"""Actor-only GRPO runtime; checkpoints contain no live or reusable group."""

import statistics
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import torch
from loguru import logger

from sequifier.config.policy_plan import GRPOPhaseConfig
from sequifier.io.atomic import write_json_atomic
from sequifier.model.frozen_image_cache import (
    frozen_image_cache_scope,
    image_cache_statistics,
    release_image_cache_entries,
)
from sequifier.rl.environment import (
    adapter_identity,
    load_online_adapter,
    require_group_start,
)
from sequifier.rl.grpo_collector import collect_groups
from sequifier.rl.grpo_update import update_grpo
from sequifier.rl.run_state import PhaseContext, PhaseSession, require_equal
from sequifier.trajectory.archives import collection_archive


def run_grpo_phase(context: PhaseContext[GRPOPhaseConfig, None]):
    actor, phase = context.actor.eval(), context.phase
    session = PhaseSession(context)
    state = session.state
    groups, env_steps = state.get("groups", 0), state.get("env_steps", 0)
    restart_stream = state.get("restart_stream", 0)
    if (
        not 0 <= groups <= phase.total_groups
        or session.transaction
        != (groups + phase.groups_per_iteration - 1) // phase.groups_per_iteration
    ):
        raise ValueError("GRPO checkpoint is not a complete update/group boundary")
    if groups != min(
        session.transaction * phase.groups_per_iteration, phase.total_groups
    ):
        raise ValueError("GRPO checkpoint group counter is not a complete iteration")
    if (
        not groups * phase.group_size
        <= env_steps
        <= groups * phase.group_size * phase.max_steps
    ):
        raise ValueError("GRPO environment clock exceeds authored bounds")
    actor.grpo_policy_version = session.transaction
    adapter = load_online_adapter(phase.environment)
    resources = ExitStack()
    workers = None
    try:
        require_group_start(adapter, exact=context.run.resume.mode == "exact")
        identity = adapter_identity(adapter)
        if state:
            require_equal(state["adapter_identity"], identity, "grpo.adapter_identity")
            if context.run.resume.mode != "exact" and groups < phase.total_groups:
                restart_stream += 1
                session.record_continuation(
                    mode="new_groups",
                    restart_stream=restart_stream,
                    trajectory_equivalent=False,
                )

        session.restore_random()
        # Register cleanup before collection or evaluation can fail. Both
        # optimizations are disabled for configurations that omit their opt-ins.
        resources.enter_context(
            frozen_image_cache_scope(
                actor, context.reference, phase.image_feature_cache_bytes
            )
        )
        if phase.simulator_workers > 1 and groups < phase.total_groups:
            from sequifier.rl.simulator_workers import (
                SimulatorWorkers,
                require_parallel_group_start,
            )

            require_parallel_group_start(adapter)
            workers = SimulatorWorkers(
                phase.environment,
                min(
                    phase.simulator_workers,
                    phase.groups_per_iteration * phase.group_size,
                ),
                phase.worker_timeout_seconds,
                phase.worker_threads,
                identity,
            )
            resources.callback(workers.close)

        if session.should_evaluate_initial():
            session.evaluate_online(adapter, initial=True)
        while groups < phase.total_groups and not session.stopping.triggered:
            count = min(phase.groups_per_iteration, phase.total_groups - groups)
            collection_start = perf_counter()
            buffer, executed = collect_groups(
                context,
                adapter,
                groups,
                count,
                session.transaction,
                env_steps,
                restart_stream,
                workers=workers,
            )
            collection_seconds = perf_counter() - collection_start
            update_start = perf_counter()
            record = update_grpo(session, buffer)
            if actor.device.type == "cuda":
                torch.cuda.synchronize(actor.device)
            update_seconds = perf_counter() - update_start
            env_steps += executed
            groups += count
            session.transaction += 1
            final = groups == phase.total_groups
            if phase.archive_directory:
                write_json_atomic(
                    Path(phase.archive_directory)
                    / f"groups-{session.transaction:08d}.json",
                    collection_archive(
                        "grpo",
                        {
                            "format": "sequifier.grpo-archive",
                            "version": 1,
                            "reference_identity": context.reference_identity,
                            "generation": buffer.generation,
                            "trajectories": [t.as_dict() for t in buffer.trajectories],
                        },
                    ),
                )
            evaluation_start = perf_counter()
            if session.should_evaluate(final):
                session.evaluate_online(adapter)
            evaluation_seconds = perf_counter() - evaluation_start
            scores = [t.score for t in buffer.trajectories]
            record.update(
                {
                    "transaction": session.transaction,
                    "groups": groups,
                    "env_steps": env_steps,
                    "policy_version": actor.grpo_policy_version,
                    "collected_policy_version": buffer.generation,
                    "score_mean": statistics.fmean(scores),
                    "score_std": statistics.pstdev(scores),
                    "zero_variance_groups": sum(
                        len({t.score for t in group}) == 1 for group in buffer.groups
                    ),
                    "lengths": [len(t.actions) for t in buffer.trajectories],
                    "cutoffs": [t.cutoff_reason for t in buffer.trajectories],
                    "start_tokens": [group[0].start_token for group in buffer.groups],
                    "buffer_bytes": buffer.estimated_bytes,
                    "simulator_workers": len(workers) if workers is not None else 1,
                    "rollout_inference_batch_size": phase.rollout_inference_batch_size,
                    "collection_seconds": collection_seconds,
                    "collection_env_steps_per_second": executed / collection_seconds,
                    "update_seconds": update_seconds,
                    "evaluation_seconds": evaluation_seconds,
                }
            )
            record.update(image_cache_statistics(actor, context.reference))
            release_image_cache_entries(actor, context.reference)
            session.record_training(final=final, **record)
            logger.info(
                "GRPO iteration {}: {} steps, collection {:.2f}s ({:.1f} steps/s), "
                "update {:.2f}s, evaluation {:.2f}s",
                session.transaction,
                executed,
                collection_seconds,
                executed / collection_seconds,
                update_seconds,
                evaluation_seconds,
            )
            session.checkpoint(
                final,
                groups=groups,
                env_steps=env_steps,
                adapter_identity=identity,
                restart_stream=restart_stream,
            )
    finally:
        try:
            resources.close()
        finally:
            adapter.close()
    return session.finish(
        {
            "optimizer_steps": session.steps["actor"],
            "groups": groups,
            "env_steps": env_steps,
        },
        {
            "reference_weight_hash": context.reference_identity,
            "adapter": identity,
        },
    )
