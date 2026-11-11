"""Fixed-dataset categorical IQL as a phase-local actor/value/twin-Q runtime."""

import math
from typing import cast

import torch

from sequifier.config.policy_plan import IQLPhaseConfig
from sequifier.io.iql_dataset import IQLBatch
from sequifier.io.tensors import collate_tensor_records
from sequifier.rl.models import make_iql_models
from sequifier.rl.offline_data import OfflineData
from sequifier.rl.run_state import PhaseContext, PhaseSession


def _select(inputs, mask):
    return type(inputs)(
        {name: value[mask] for name, value in inputs.inputs.items()},
        {name: value[mask] for name, value in inputs.metadata.items()},
    )


def _polyak(target, source, rate):
    with torch.no_grad():
        for target_parameter, parameter in zip(
            target.parameters(), source.parameters()
        ):
            target_parameter.lerp_(parameter, rate)
        for target_buffer, buffer in zip(target.buffers(), source.buffers()):
            target_buffer.copy_(buffer)


def _evaluate(actor, value, q1, q2, dataset, batch_size, gamma, device):
    totals = {"action_nll": 0.0, "bellman_residual": 0.0, "value": 0.0, "q_min": 0.0}
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = cast(
                IQLBatch,
                collate_tensor_records(
                    [
                        dataset[index]
                        for index in range(start, min(start + batch_size, len(dataset)))
                    ]
                ),
            ).to(device)
            action = batch.action
            v = value(batch.current)
            q = torch.minimum(q1(batch.current, action), q2(batch.current, action))
            nll = -actor.evaluate_actions(batch.current, action)
            successor_value = torch.zeros_like(v)
            if batch.bootstrap.any():
                successor_value[batch.bootstrap] = value(
                    _select(batch.successor, batch.bootstrap)
                )
            totals["action_nll"] += float(nll.sum())
            totals["bellman_residual"] += float(
                (q - batch.reward - gamma * successor_value).abs().sum()
            )
            totals["value"] += float(v.sum())
            totals["q_min"] += float(q.sum())
    result = {key: value / len(dataset) for key, value in totals.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("non-finite IQL held-out likelihood or critic diagnostics")
    return result


def run_iql_phase(context: PhaseContext[IQLPhaseConfig, OfflineData]):
    actor, phase = context.actor.eval(), context.phase
    data = context.data.prepare_transitions()
    value, q1, q2, targets = make_iql_models(
        actor.execution_config,
        actor.interface_name,
        actor.config.actions,
    )
    for module in (value, q1, q2, *targets):
        module.eval()
    session = PhaseSession(
        context,
        modules={"value": value, "critic": torch.nn.ModuleList([q1, q2])},
        frozen={"target_q1": targets[0], "target_q2": targets[1]},
    )
    if not 0 <= session.transaction <= phase.iterations:
        raise ValueError("IQL checkpoint exceeds phase update budget")
    if any(steps != session.transaction for steps in session.steps.values()):
        raise ValueError("IQL optimizer clocks disagree with committed updates")
    for update in range(session.transaction + 1, phase.iterations + 1):
        indices = torch.randint(
            len(data[0]), (phase.batch_size,), generator=session.sampler
        )
        batch = cast(
            IQLBatch, collate_tensor_records([data[0][int(i)] for i in indices])
        ).to(actor.device)
        with torch.no_grad():
            qbar = torch.minimum(
                targets[0](batch.current, batch.action),
                targets[1](batch.current, batch.action),
            )
            old_v = value(batch.current)
            advantage = qbar - old_v
            weight = torch.exp(
                (phase.inverse_temperature * advantage).clamp(
                    max=math.log(phase.max_advantage_weight)
                )
            )
            successor_value = torch.zeros_like(old_v)
            if batch.bootstrap.any():
                successor_value[batch.bootstrap] = value(
                    _select(batch.successor, batch.bootstrap)
                )
            target = batch.reward + phase.gamma * successor_value
        delta = qbar - value(batch.current)
        v_loss = (
            torch.where(delta < 0, 1 - phase.expectile, phase.expectile)
            * delta.square()
        ).mean()
        q_loss = (
            (q1(batch.current, batch.action) - target).square()
            + (q2(batch.current, batch.action) - target).square()
        ).mean()
        actor_loss = -(
            weight * actor.evaluate_actions(batch.current, batch.action)
        ).mean()
        losses = {"value": v_loss, "critic": q_loss, "actor": actor_loss}
        session.zero_grad()
        for role, loss in losses.items():
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite IQL {role} objective")
            loss.backward()
        session.step()
        _polyak(targets[0], q1, phase.target_update_rate)
        _polyak(targets[1], q2, phase.target_update_rate)
        session.transaction = update
        final = update == phase.iterations
        if session.should_evaluate(final):
            held_out = _evaluate(
                actor,
                value,
                q1,
                q2,
                data[1],
                phase.batch_size,
                phase.gamma,
                actor.device,
            )
            session.record_evaluation(
                held_out["action_nll"],
                **{
                    f"{role}_loss": float(loss.detach())
                    for role, loss in losses.items()
                },
                **{f"validation_{key}": val for key, val in held_out.items()},
                advantage_mean=float(advantage.mean()),
                weight_mean=float(weight.mean()),
                weight_cap_fraction=float(
                    (weight >= phase.max_advantage_weight).float().mean()
                ),
            )
        session.checkpoint(final)
    return session.finish(
        {f"{role}_optimizer_steps": steps for role, steps in session.steps.items()},
        {"data": context.data.contract},
    )
