"""Actor-only GRPO runtime; checkpoints contain no live or reusable group."""

import statistics
from copy import deepcopy
from pathlib import Path

import torch

from sequifier.config.policy_plan import GRPOPhaseConfig
from sequifier.io.atomic import write_json_atomic
from sequifier.rl.environment import (
    adapter_identity,
    load_online_adapter,
    require_group_start,
)
from sequifier.rl.grpo_collector import collect_groups
from sequifier.rl.grpo_update import update_grpo
from sequifier.rl.ppo_evaluation import evaluate_policy
from sequifier.rl.run_state import PhaseContext, PhaseSession, require_equal


def run_grpo_phase(context: PhaseContext[GRPOPhaseConfig, None]):
    actor, phase = context.actor.eval(), context.phase
    session = PhaseSession(context)
    state = session.state
    groups, env_steps = state.get("groups", 0), state.get("env_steps", 0)
    evaluations, continuations = (
        list(state.get("evaluations", [])),
        list(state.get("continuations", [])),
    )
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
    try:
        require_group_start(adapter, exact=context.run.resume.mode == "exact")
        identity = adapter_identity(adapter)
        if state:
            require_equal(state["adapter_identity"], identity, "grpo.adapter_identity")
            if context.run.resume.mode != "exact" and groups < phase.total_groups:
                restart_stream += 1
                continuations.append(
                    {
                        "transaction": session.transaction,
                        "mode": "new_groups",
                        "restart_stream": restart_stream,
                        "trajectory_equivalent": False,
                    }
                )

        session.restore_random()

        def evaluate():
            result = evaluate_policy(actor, phase.evaluation, forbidden_adapter=adapter)
            score = float(
                torch.tensor(
                    result["score_mean"] * phase.reward_scale, dtype=torch.float32
                )
            )
            if not torch.isfinite(torch.tensor(score)):
                raise ValueError("GRPO evaluation score exceeds finite FP32 range")
            session.observe(score)
            evaluations.append(
                {**result, "score_mean": score, "transaction": session.transaction}
            )

        while groups < phase.total_groups:
            count = min(phase.groups_per_iteration, phase.total_groups - groups)
            buffer, executed = collect_groups(
                context,
                adapter,
                groups,
                count,
                session.transaction,
                env_steps,
                restart_stream,
            )
            record = update_grpo(session, buffer)
            env_steps += executed
            groups += count
            session.transaction += 1
            final = groups == phase.total_groups
            if phase.archive_directory:
                write_json_atomic(
                    Path(phase.archive_directory)
                    / f"groups-{session.transaction:08d}.json",
                    {
                        "format": "sequifier.grpo-archive",
                        "version": 1,
                        "reference_identity": context.reference_identity,
                        "generation": buffer.generation,
                        "trajectories": [t.as_dict() for t in buffer.trajectories],
                    },
                )
            if session.should_evaluate(final):
                evaluate()
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
                    "start_tokens": [
                        deepcopy(group[0].start_token) for group in buffer.groups
                    ],
                    "buffer_bytes": buffer.estimated_bytes,
                }
            )
            session.metrics.append(record)
            session.checkpoint(
                final,
                groups=groups,
                env_steps=env_steps,
                adapter_identity=identity,
                evaluations=evaluations,
                continuations=continuations,
                restart_stream=restart_stream,
            )
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
            "evaluations": evaluations,
            "continuations": continuations,
        },
    )
