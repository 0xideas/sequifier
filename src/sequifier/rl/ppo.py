"""Phase-local PPO lifecycle over the shared collector and clipped actor/V update."""

from copy import deepcopy
from pathlib import Path

from sequifier.config.policy_plan import PPOPhaseConfig
from sequifier.rl.environment import (
    adapter_identity,
    load_online_adapter,
    require_exact_snapshot,
)
from sequifier.rl.models import HistoryCritic
from sequifier.rl.ppo_collector import PPOCollector
from sequifier.rl.ppo_evaluation import evaluate_policy
from sequifier.rl.ppo_update import update_ppo
from sequifier.rl.run_state import PhaseContext, PhaseSession, require_equal
from sequifier.rl.snapshots import snapshots_equal


def run_ppo_phase(context: PhaseContext[PPOPhaseConfig, None]):
    actor, phase = context.actor.eval(), context.phase
    value = HistoryCritic(
        deepcopy(actor.execution_config), actor.interface_name, None
    ).eval()
    session = PhaseSession(context, modules={"value": value})
    state = session.state
    env_steps = state.get("env_steps", 0)
    evaluations = list(state.get("evaluations", []))
    continuations = list(state.get("continuations", []))
    if (
        env_steps
        != min(session.transaction * phase.rollout_env_steps, phase.total_env_steps)
        or not 0 <= env_steps <= phase.total_env_steps
    ):
        raise ValueError("PPO checkpoint is not a complete collection/update boundary")
    actor.ppo_policy_version = session.transaction
    adapter = load_online_adapter(phase.environment)
    try:
        if context.run.resume.mode == "exact":
            require_exact_snapshot(adapter)
        identity = adapter_identity(adapter)
        collector = PPOCollector(
            actor, value, adapter, phase, context.run.seed + context.index
        )
        if state:
            require_equal(state["adapter_identity"], identity, "ppo.adapter_identity")
            exact = context.run.resume.mode == "exact"
            if exact:
                if state["environment_state"] is None:
                    raise ValueError(
                        "exact PPO resume requires an embedded environment snapshot"
                    )
                adapter.restore_state(deepcopy(state["environment_state"]))
                if not snapshots_equal(
                    adapter.snapshot_state(), state["environment_state"]
                ):
                    raise ValueError("PPO adapter did not restore its snapshot exactly")
            collector.load_state_dict(state["collector"], exact=exact)
            continuations.append(
                {
                    "transaction": session.transaction,
                    "mode": context.run.resume.mode,
                    "trajectory_equivalent": exact,
                }
            )
        session.restore_random()
        while env_steps < phase.total_env_steps:
            horizon = min(phase.rollout_env_steps, phase.total_env_steps - env_steps)
            buffer, episodes = collector.collect(horizon, session.transaction)
            if phase.archive_directory:
                buffer.archive(
                    Path(phase.archive_directory)
                    / f"collection-{session.transaction+1:08d}.json",
                    include_scores=phase.archive_scores,
                )
            record = update_ppo(session, value, buffer)
            env_steps += horizon
            session.transaction += 1
            final = env_steps == phase.total_env_steps
            if session.should_evaluate(final):
                result = evaluate_policy(
                    actor, phase.evaluation, forbidden_adapter=adapter
                )
                session.observe(result["selection_value"])
                evaluations.append(
                    {
                        **result,
                        "transaction": session.transaction,
                        "env_steps": env_steps,
                    }
                )
            session.metrics.append(
                {
                    **record,
                    "transaction": session.transaction,
                    "env_steps": env_steps,
                    "policy_version": actor.ppo_policy_version,
                    "episodes": episodes,
                }
            )
            if session.should_checkpoint(final):
                environment_state = None
                if adapter.capabilities.exact_snapshot:
                    require_exact_snapshot(adapter)
                    environment_state = adapter.snapshot_state()
                    if not isinstance(environment_state, dict) or not snapshots_equal(
                        environment_state, environment_state
                    ):
                        raise ValueError(
                            "PPO snapshot must be a finite portable state dictionary"
                        )
                session.checkpoint(
                    final,
                    env_steps=env_steps,
                    evaluations=evaluations,
                    continuations=continuations,
                    collector=collector.state_dict(),
                    environment_state=environment_state,
                    adapter_identity=identity,
                )
    finally:
        adapter.close()
    return session.finish(
        {
            "env_steps": env_steps,
            **{
                f"{role}_optimizer_steps": steps
                for role, steps in session.steps.items()
            },
        },
        {
            "adapter": identity,
            "evaluations": evaluations,
            "continuations": continuations,
        },
    )
