"""Phase-local PPO lifecycle over the shared collector and clipped actor/V update."""

from copy import deepcopy
from pathlib import Path

import torch

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
from sequifier.rl.run_state import (
    PhaseContext,
    Selection,
    actor_state,
    clone_weights,
    make_optimization,
    require_equal,
    restore_actor_state,
    restore_rng_state,
    validate_finite_state,
)
from sequifier.rl.snapshots import snapshots_equal


def run_ppo_phase(context: PhaseContext[PPOPhaseConfig, None]):
    actor, phase = context.actor.eval(), context.phase
    value = HistoryCritic(
        deepcopy(actor.execution_config), actor.interface_name, None
    ).eval()
    optimizer, scheduler = make_optimization(actor, phase)
    value_optimizer, value_scheduler = make_optimization(value, phase, "value")
    sampler = torch.Generator().manual_seed(context.run.seed + context.index)
    state = context.resume_state or {}
    transaction, env_steps = state.get("transaction", 0), state.get("env_steps", 0)
    metrics, evaluations = (
        list(state.get("metrics", [])),
        list(state.get("evaluations", [])),
    )
    continuations = list(state.get("continuations", []))
    clocks = dict(state.get("clocks", {"actor": 0, "value": 0}))
    if (
        env_steps != min(transaction * phase.rollout_length, phase.total_env_steps)
        or not 0 <= env_steps <= phase.total_env_steps
    ):
        raise ValueError("PPO checkpoint is not a complete collection/update boundary")
    if state:
        for key in ("value", "value_optimizer", "value_scheduler"):
            validate_finite_state(state[key], f"ppo.{key}")
        if any(
            not isinstance(clock, int)
            or isinstance(clock, bool)
            or not transaction <= clock <= phase.optimizer_step_budget()
            for clock in clocks.values()
        ) or set(clocks) != {"actor", "value"}:
            raise ValueError("PPO optimizer clocks disagree with committed progress")
        value.load_state_dict(state["value"], strict=True)
        value_optimizer.load_state_dict(state["value_optimizer"])
        value_scheduler.load_state_dict(state["value_scheduler"])
    restore_actor_state(context, optimizer, scheduler, sampler)
    selection = Selection(
        phase,
        phase.evaluation.metric,
        phase.evaluation.direction,
        state.get("selection"),
    )
    actor.ppo_policy_version = transaction
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
                    "transaction": transaction,
                    "mode": context.run.resume.mode,
                    "trajectory_equivalent": exact,
                }
            )
        if state:
            restore_rng_state(actor.device, state["random"])
        while env_steps < phase.total_env_steps:
            horizon = min(phase.rollout_length, phase.total_env_steps - env_steps)
            buffer, episodes = collector.collect(horizon, transaction)
            if phase.archive_directory:
                buffer.archive(
                    Path(phase.archive_directory)
                    / f"collection-{transaction+1:08d}.json",
                    include_scores=phase.archive_scores,
                )
            record = update_ppo(
                actor,
                value,
                buffer,
                phase,
                {"actor": optimizer, "value": value_optimizer},
                {"actor": scheduler, "value": value_scheduler},
                sampler,
                transaction,
                clocks,
            )
            env_steps, transaction = env_steps + horizon, transaction + 1
            if (
                transaction % phase.evaluation_interval == 0
                or env_steps == phase.total_env_steps
            ):
                result = evaluate_policy(
                    actor, phase.evaluation, forbidden_adapter=adapter
                )
                selection.observe(actor, result["selection_value"], transaction)
                evaluations.append(
                    {**result, "transaction": transaction, "env_steps": env_steps}
                )
            metrics.append(
                {
                    **record,
                    "transaction": transaction,
                    "env_steps": env_steps,
                    "policy_version": actor.ppo_policy_version,
                    "episodes": episodes,
                }
            )
            if (
                transaction % phase.checkpoint_interval == 0
                or env_steps == phase.total_env_steps
            ):
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
                context.commit(
                    actor_state(
                        context,
                        optimizer,
                        scheduler,
                        sampler,
                        selection,
                        transaction,
                        metrics,
                        value=clone_weights(value),
                        value_optimizer=value_optimizer.state_dict(),
                        value_scheduler=value_scheduler.state_dict(),
                        clocks=clocks,
                        env_steps=env_steps,
                        evaluations=evaluations,
                        continuations=continuations,
                        collector=collector.state_dict(),
                        environment_state=environment_state,
                        adapter_identity=identity,
                    )
                )
    finally:
        adapter.close()
    return selection.finish(
        context,
        transaction,
        {
            "env_steps": env_steps,
            **{f"{key}_optimizer_steps": val for key, val in clocks.items()},
        },
        metrics,
        {
            "adapter": identity,
            "evaluations": evaluations,
            "continuations": continuations,
        },
    )
