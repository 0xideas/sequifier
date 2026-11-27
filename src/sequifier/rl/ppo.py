"""Phase-local PPO lifecycle over the shared collector and clipped actor/V update."""

from copy import deepcopy
from dataclasses import replace
from time import perf_counter

import torch

from sequifier.config.policy_plan import PPOPhaseConfig
from sequifier.rl.collection import stream_seed
from sequifier.rl.environment import (
    adapter_identity,
    load_online_adapter,
    require_exact_snapshot,
)
from sequifier.rl.ppo_collector import PPOCollector
from sequifier.rl.ppo_update import update_ppo
from sequifier.rl.run_state import PhaseContext, PhaseSession, require_equal
from sequifier.rl.sequence_output import CollectedSequences
from sequifier.rl.snapshots import snapshots_equal
from sequifier.runtime.resource_accounting import allocation


def build_ppo_roles(actor, phase, *, metadata_only, restoring):
    from sequifier.rl.models import HistoryCritic, critic_history_options
    from sequifier.rl.phase_optimization import PhaseRoles

    history = critic_history_options(
        actor, phase, metadata_only=metadata_only, restoring=restoring
    )
    value = HistoryCritic.ppo_value(actor, **history).eval()
    return PhaseRoles({"actor": actor.network, "value": value}, {}, {"value": ["head"]})


def validate_ppo_adapter(adapter, phase, exact):
    if exact:
        require_exact_snapshot(adapter)


def run_ppo_phase(context: PhaseContext[PPOPhaseConfig, None]):
    actor, phase = context.actor.eval(), context.phase
    session = PhaseSession(context)
    value = session.calls["value"]
    strategy = session.strategy
    distributed = strategy.world_size > 1
    state = session.state
    sequences = CollectedSequences(context, state)
    env_steps = state.get("env_steps", 0)
    if (
        env_steps
        != min(
            session.transaction * phase.collection.env_steps_per_iteration,
            phase.budget.env_steps,
        )
        or not 0 <= env_steps <= phase.budget.env_steps
    ):
        raise ValueError("PPO checkpoint is not a complete collection/update boundary")
    actor.ppo_policy_version = session.transaction
    adapter = load_online_adapter(phase.collection.environment)
    collector = None
    try:
        validate_ppo_adapter(adapter, phase, context.run.resume.mode == "exact")
        identity = adapter_identity(
            adapter, context.adapter_contract[f"{context.index}.training"]
        )
        collector = PPOCollector(
            actor,
            value,
            adapter,
            phase,
            stream_seed(context.identity.sampling_seed, "ppo-rank", strategy.rank)
            if distributed
            else context.identity.sampling_seed,
            stream=strategy.rank if distributed else None,
        )
        if state:
            require_equal(state["adapter_identity"], identity, "ppo.adapter_identity")
            exact = context.run.resume.mode == "exact"
            if exact and state["collector"] is not None:
                if state["environment_state"] is None:
                    raise ValueError(
                        "exact PPO resume requires an embedded environment snapshot"
                    )
                adapter.restore_state(deepcopy(state["environment_state"]))
                if not snapshots_equal(
                    adapter.snapshot_state(), state["environment_state"]
                ):
                    raise ValueError("PPO adapter did not restore its snapshot exactly")
            if state["collector"] is not None:
                collector.load_state_dict(state["collector"], exact=exact)
            session.record_continuation(
                mode=context.run.resume.mode, trajectory_equivalent=exact
            )

        def checkpoint_payload():
            environment_state = None
            if collector.reset_count and adapter.capabilities.exact_snapshot:
                require_exact_snapshot(adapter)
                environment_state = adapter.snapshot_state()
                if not isinstance(environment_state, dict) or not snapshots_equal(
                    environment_state, environment_state
                ):
                    raise ValueError(
                        "PPO snapshot must be a finite portable state dictionary"
                    )
            return {
                "env_steps": env_steps,
                "per_decision_entropy": state.get("per_decision_entropy", not state),
                "collector": collector.state_dict() if collector.reset_count else None,
                **sequences.checkpoint(),
                "environment_state": environment_state,
                "adapter_identity": identity,
            }

        session.restore_random()
        session.evaluate_initial(session.evaluate_online, adapter)
        while env_steps < phase.budget.env_steps and not session.stopping.triggered:
            horizon = min(
                phase.collection.env_steps_per_iteration,
                phase.budget.env_steps - env_steps,
            )
            with allocation("rollout_storage") as storage:
                collection_start = perf_counter()
                local_horizon = len(range(strategy.rank, horizon, strategy.world_size))
                with session.rounds.scope():
                    buffer, episodes = (
                        collector.collect(
                            local_horizon,
                            session.transaction,
                            storage=storage,
                            normalize=False if distributed else None,
                        )
                        if local_horizon
                        else (None, [])
                    )
                if distributed and phase.normalize_advantages and horizon > 1:
                    advantages = torch.tensor(
                        buffer.advantages if buffer else (), dtype=torch.float64
                    )
                    mean = strategy.sum(float(advantages.sum())) / horizon
                    variance = (
                        strategy.sum(float((advantages - mean).square().sum()))
                        / horizon
                    )
                    if buffer is not None:
                        buffer = replace(
                            buffer,
                            advantages=tuple(
                                ((advantages - mean) / max(variance**0.5, 1e-8))
                                .float()
                                .tolist()
                            ),
                        )
                episodes = [
                    item
                    for stream in strategy.gather_for_writer(episodes)
                    for item in stream
                ]
                session.collection_seconds += perf_counter() - collection_start
                sequences.record(buffer)
                update_start = perf_counter()
                record = update_ppo(session, value, buffer)
                session.update_seconds += perf_counter() - update_start
                del buffer
            env_steps += horizon
            session.complete_update(
                online=True,
                final=env_steps == phase.budget.env_steps,
                training={
                    **record,
                    "env_steps": env_steps,
                    "policy_version": actor.ppo_policy_version,
                    "episodes": episodes,
                },
                evaluate=lambda: session.evaluate_online(adapter, env_steps=env_steps),
                checkpoint=checkpoint_payload,
            )
    finally:
        if collector is not None:
            collector.cursor.close()
        adapter.close()
    return session.finish(
        {
            "env_steps": env_steps,
            **session.optimizer_clocks(),
        },
        {"adapter": identity, **sequences.publish()},
    )
