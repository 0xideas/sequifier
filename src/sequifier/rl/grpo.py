"""Actor-only GRPO runtime; checkpoints contain no live or reusable group."""

import statistics
from copy import deepcopy
from pathlib import Path

import torch

from sequifier.config.policy_plan import GRPOPhaseConfig
from sequifier.rl.environment import (
    adapter_identity,
    load_online_adapter,
    require_group_start,
)
from sequifier.rl.grpo_collector import collect_groups
from sequifier.rl.grpo_update import update_grpo
from sequifier.rl.ppo_evaluation import evaluate_policy
from sequifier.rl.run_state import (
    PhaseContext,
    Selection,
    actor_state,
    require_equal,
    restore_actor_state,
    restore_rng_state,
    write_json_atomic,
)


def run_grpo_phase(context: PhaseContext[GRPOPhaseConfig, None]):
    actor, phase = context.actor.eval(), context.phase
    optimizer, scheduler = context.optimization["actor"].build(actor)
    sampler = torch.Generator().manual_seed(context.run.seed + context.index)
    restore_actor_state(context, optimizer, scheduler)
    if context.resume_state is not None:
        sampler.set_state(context.resume_state["sampler"])
    state = context.resume_state or {}
    transaction, metrics = state.get("transaction", 0), list(state.get("metrics", []))
    groups, env_steps, optimizer_steps = (
        state.get("groups", 0),
        state.get("env_steps", 0),
        state.get("optimizer_steps", 0),
    )
    evaluations, continuations = (
        list(state.get("evaluations", [])),
        list(state.get("continuations", [])),
    )
    restart_stream = state.get("restart_stream", 0)
    if (
        not 0 <= groups <= phase.total_groups
        or transaction
        != (groups + phase.groups_per_update - 1) // phase.groups_per_update
    ):
        raise ValueError("GRPO checkpoint is not a complete update/group boundary")
    if groups != min(transaction * phase.groups_per_update, phase.total_groups):
        raise ValueError("GRPO checkpoint group counter is not a complete iteration")
    if (
        not transaction <= optimizer_steps <= phase.optimizer_step_budget()
        or not groups * phase.group_size
        <= env_steps
        <= groups * phase.group_size * phase.max_steps
    ):
        raise ValueError(
            "GRPO checkpoint optimizer/environment clocks exceed authored bounds"
        )
    selection = Selection(phase, "score_mean", "max", state.get("selection"))
    actor.grpo_policy_version = transaction
    adapter = load_online_adapter(phase.environment)
    try:
        require_group_start(adapter, exact=context.run.resume.mode == "exact")
        identity = adapter_identity(adapter)
        if state:
            require_equal(state["adapter_identity"], identity, "grpo.adapter_identity")
            if context.run.resume.mode != "exact" and groups < phase.total_groups:
                restart_stream += 1
                continuations.append(
                    {
                        "transaction": transaction,
                        "mode": "new_groups",
                        "restart_stream": restart_stream,
                        "trajectory_equivalent": False,
                    }
                )

        if state:
            restore_rng_state(actor.device, state["random"])

        def evaluate():
            result = evaluate_policy(actor, phase.evaluation, forbidden_adapter=adapter)
            score = float(
                torch.tensor(
                    result["score_mean"] * phase.reward_scale, dtype=torch.float32
                )
            )
            if not torch.isfinite(torch.tensor(score)):
                raise ValueError("GRPO evaluation score exceeds finite FP32 range")
            selection.observe(actor, score, transaction)
            evaluations.append(
                {**result, "score_mean": score, "transaction": transaction}
            )

        while groups < phase.total_groups:
            count = min(phase.groups_per_update, phase.total_groups - groups)
            buffer, executed = collect_groups(
                context, adapter, groups, count, transaction, env_steps, restart_stream
            )
            record = update_grpo(
                context,
                buffer,
                optimizer,
                scheduler,
                sampler,
                transaction,
                optimizer_steps,
            )
            optimizer_steps = record["optimizer_steps"]
            env_steps, groups, transaction = (
                env_steps + executed,
                groups + count,
                transaction + 1,
            )
            if phase.archive_directory:
                write_json_atomic(
                    Path(phase.archive_directory) / f"groups-{transaction:08d}.json",
                    {
                        "format": "sequifier.grpo-archive",
                        "version": 1,
                        "reference_identity": context.reference_identity,
                        "generation": buffer.generation,
                        "trajectories": [t.as_dict() for t in buffer.trajectories],
                    },
                )
            if (
                transaction % phase.evaluation_interval == 0
                or groups == phase.total_groups
            ):
                evaluate()
            scores = [t.score for t in buffer.trajectories]
            record.update(
                {
                    "transaction": transaction,
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
                    "start_tokens": [
                        deepcopy(group[0].start_token) for group in buffer.groups
                    ],
                    "buffer_bytes": buffer.estimated_bytes,
                }
            )
            metrics.append(record)
            if (
                transaction % phase.checkpoint_interval == 0
                or groups == phase.total_groups
            ):
                context.commit(
                    actor_state(
                        context,
                        optimizer,
                        scheduler,
                        selection,
                        transaction,
                        metrics,
                        sampler=sampler.get_state(),
                        groups=groups,
                        env_steps=env_steps,
                        optimizer_steps=optimizer_steps,
                        adapter_identity=identity,
                        evaluations=evaluations,
                        continuations=continuations,
                        restart_stream=restart_stream,
                    )
                )
    finally:
        adapter.close()
    return selection.finish(
        context,
        transaction,
        {"optimizer_steps": optimizer_steps, "groups": groups, "env_steps": env_steps},
        metrics,
        {
            "reference_weight_hash": context.reference_identity,
            "adapter": identity,
            "evaluations": evaluations,
            "continuations": continuations,
        },
    )
