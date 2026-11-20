"""Phase-local PPO lifecycle over the shared collector and clipped actor/V update."""

from copy import deepcopy
from pathlib import Path

from sequifier.config.policy_plan import PPOPhaseConfig
from sequifier.rl.environment import (
    adapter_identity,
    load_online_adapter,
    require_exact_snapshot,
)
from sequifier.rl.ppo_collector import PPOCollector
from sequifier.rl.ppo_update import update_ppo
from sequifier.rl.run_state import PhaseContext, PhaseSession, require_equal
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
    value = session.modules["value"]
    state = session.state
    env_steps = state.get("env_steps", 0)
    if (
        env_steps
        != min(session.transaction * phase.rollout_env_steps, phase.total_env_steps)
        or not 0 <= env_steps <= phase.total_env_steps
    ):
        raise ValueError("PPO checkpoint is not a complete collection/update boundary")
    actor.ppo_policy_version = session.transaction
    adapter = load_online_adapter(phase.environment)
    collector = None
    try:
        validate_ppo_adapter(adapter, phase, context.run.resume.mode == "exact")
        identity = adapter_identity(
            adapter, context.adapter_contract[f"{context.index}.training"]
        )
        collector = PPOCollector(
            actor, value, adapter, phase, context.identity.sampling_seed
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
            session.record_continuation(
                mode=context.run.resume.mode, trajectory_equivalent=exact
            )
        session.restore_random()
        if session.should_evaluate_initial():
            session.evaluate_online(adapter, initial=True)
        while env_steps < phase.total_env_steps and not session.stopping.triggered:
            horizon = min(phase.rollout_env_steps, phase.total_env_steps - env_steps)
            with allocation("rollout_storage") as storage:
                buffer, episodes = collector.collect(
                    horizon, session.transaction, storage=storage
                )
                if phase.archive_directory:
                    with allocation("rollout_archive") as archive:
                        archive.resize(buffer.estimated_bytes)
                        buffer.archive(
                            Path(phase.archive_directory)
                            / f"collection-{session.transaction + 1:08d}.json",
                            include_scores=phase.archive_scores,
                        )
                record = update_ppo(session, value, buffer)
                del buffer
            env_steps += horizon
            session.transaction += 1
            final = env_steps == phase.total_env_steps
            if session.should_evaluate(final):
                session.evaluate_online(adapter, env_steps=env_steps)
            session.record_training(
                final=final,
                **{
                    **record,
                    "transaction": session.transaction,
                    "env_steps": env_steps,
                    "policy_version": actor.ppo_policy_version,
                    "episodes": episodes,
                },
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
                    collector=collector.state_dict(),
                    environment_state=environment_state,
                    adapter_identity=identity,
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
        {"adapter": identity},
    )
