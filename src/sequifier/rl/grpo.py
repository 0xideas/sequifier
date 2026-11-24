"""Actor-only GRPO runtime; checkpoints contain no live or reusable group."""

import statistics
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import torch
from loguru import logger

from sequifier.config.policy_plan import GRPOPhaseConfig
from sequifier.contract_versions import ContractVersions
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
from sequifier.rl.sequence_output import CollectedSequences
from sequifier.runtime.resource_accounting import allocation
from sequifier.trajectory.archives import collection_archive


def validate_grpo_adapter(adapter, phase, exact):
    require_group_start(adapter, exact=exact)
    if phase.collection.parallel is not None:
        from sequifier.rl.simulator_workers import require_parallel_group_start

        require_parallel_group_start(adapter)


def scale_evaluation_score(result, phase):
    if phase.evaluation.monitor.metric == "score_mean":
        score = torch.tensor(
            result["score_mean"] * phase.reward_scale, dtype=torch.float32
        )
        if not torch.isfinite(score):
            raise ValueError("GRPO evaluation score exceeds finite FP32 range")
        result["score_mean"] = result["selection_value"] = float(score)


def run_grpo_phase(context: PhaseContext[GRPOPhaseConfig, None]):
    actor, phase = context.actor.eval(), context.phase
    session = PhaseSession(context)
    state = session.state
    sequences = CollectedSequences(context, state)
    groups, env_steps = state.get("groups", 0), state.get("env_steps", 0)
    restart_stream = state.get("restart_stream", 0)
    if (
        not 0 <= groups <= phase.budget.groups
        or session.transaction
        != (groups + phase.collection.groups_per_iteration - 1)
        // phase.collection.groups_per_iteration
    ):
        raise ValueError("GRPO checkpoint is not a complete update/group boundary")
    if groups != min(
        session.transaction * phase.collection.groups_per_iteration, phase.budget.groups
    ):
        raise ValueError("GRPO checkpoint group counter is not a complete iteration")
    if (
        not groups * phase.collection.group_size
        <= env_steps
        <= groups * phase.collection.group_size * phase.collection.max_steps
    ):
        raise ValueError("GRPO environment clock exceeds authored bounds")
    actor.grpo_policy_version = session.transaction
    adapter = load_online_adapter(phase.collection.environment)
    resources = ExitStack()
    workers = None
    try:
        validate_grpo_adapter(adapter, phase, context.run.resume.mode == "exact")
        identity = adapter_identity(
            adapter, context.adapter_contract[f"{context.index}.training"]
        )
        if state:
            require_equal(state["adapter_identity"], identity, "grpo.adapter_identity")
            if context.run.resume.mode != "exact" and groups < phase.budget.groups:
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
                actor, context.reference, phase.collection.image_feature_cache_bytes
            )
        )
        parallel = phase.collection.parallel
        if parallel is not None and groups < phase.budget.groups:
            from sequifier.rl.simulator_workers import SimulatorWorkers

            workers = SimulatorWorkers(
                phase.collection.environment,
                min(
                    parallel.workers,
                    phase.collection.groups_per_iteration * phase.collection.group_size,
                ),
                parallel.timeout_seconds,
                parallel.threads_per_worker,
                identity,
            )
            resources.callback(workers.close)

        session.evaluate_initial(
            session.evaluate_online, adapter, adjust=scale_evaluation_score
        )
        while groups < phase.budget.groups and not session.stopping.triggered:
            count = min(
                phase.collection.groups_per_iteration, phase.budget.groups - groups
            )
            with allocation("rollout_storage") as storage:
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
                    storage=storage,
                )
                collection_seconds = perf_counter() - collection_start
                update_start = perf_counter()
                sequences.grpo(buffer)
                record = update_grpo(session, buffer)
                if actor.device.type == "cuda":
                    torch.cuda.synchronize(actor.device)
                update_seconds = perf_counter() - update_start
                env_steps += executed
                groups += count
                if phase.collection.archive_directory:
                    with allocation("rollout_archive") as archive:
                        archive.resize(buffer.estimated_bytes)
                        write_json_atomic(
                            Path(phase.collection.archive_directory)
                            / f"groups-{actor.grpo_policy_version:08d}.json",
                            collection_archive(
                                "grpo",
                                {
                                    "format": "sequifier.grpo-archive",
                                    "version": ContractVersions.GRPO_ARCHIVE,
                                    "reference_identity": context.reference_identity,
                                    "generation": buffer.generation,
                                    "trajectories": [
                                        t.as_dict() for t in buffer.trajectories
                                    ],
                                },
                            ),
                        )
                scores = [t.score for t in buffer.trajectories]
                record.update(
                    {
                        "groups": groups,
                        "env_steps": env_steps,
                        "policy_version": actor.grpo_policy_version,
                        "collected_policy_version": buffer.generation,
                        "score_mean": statistics.fmean(scores),
                        "score_std": statistics.pstdev(scores),
                        "zero_variance_groups": sum(
                            len({t.score for t in group}) == 1
                            for group in buffer.groups
                        ),
                        "lengths": [len(t.actions) for t in buffer.trajectories],
                        "cutoffs": [t.cutoff_reason for t in buffer.trajectories],
                        "start_tokens": [
                            group[0].start_token for group in buffer.groups
                        ],
                        "buffer_bytes": buffer.estimated_bytes,
                        "simulator_workers": len(workers) if workers is not None else 1,
                        "rollout_inference_batch_size": parallel.inference_batch_size
                        if parallel
                        else 1,
                        "collection_seconds": collection_seconds,
                        "collection_env_steps_per_second": (
                            executed / collection_seconds
                        ),
                        "update_seconds": update_seconds,
                    }
                )
                del buffer, scores
            evaluation_start = perf_counter()

            def training_metrics():
                record["evaluation_seconds"] = perf_counter() - evaluation_start
                record.update(image_cache_statistics(actor, context.reference))
                release_image_cache_entries(actor, context.reference)
                return record

            session.complete_online_update(
                final=groups == phase.budget.groups,
                training=training_metrics,
                evaluate=lambda: session.evaluate_online(
                    adapter, adjust=scale_evaluation_score
                ),
                checkpoint=lambda: dict(
                    groups=groups,
                    env_steps=env_steps,
                    adapter_identity=identity,
                    restart_stream=restart_stream,
                    **sequences.checkpoint(),
                ),
            )
            logger.debug(
                "GRPO iteration {}: {} steps, collection {:.2f}s ({:.1f} steps/s), "
                "update {:.2f}s, evaluation {:.2f}s",
                session.transaction,
                executed,
                collection_seconds,
                executed / collection_seconds,
                update_seconds,
                record["evaluation_seconds"],
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
            **sequences.publish(),
            "reference_weight_hash": context.reference_identity,
            "adapter": identity,
        },
    )
