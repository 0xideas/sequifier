"""Fixed-dataset categorical IQL as a phase-local actor/value/twin-Q runtime."""

import math
from typing import cast

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from sequifier.artifacts.state_dict import canonicalize_state_dict
from sequifier.config.policy_plan import IQLPhaseConfig
from sequifier.io.iql_dataset import IQLTransitionDataset
from sequifier.io.tensors import gather_tensor
from sequifier.rl.offline_data import OfflineData
from sequifier.rl.run_state import PhaseContext, PhaseSession
from sequifier.rl.window_storage import collated_windows


class TwinQ(nn.ModuleList):
    def forward(self, inputs, actions):
        return self[0](inputs, actions), self[1](inputs, actions)


def build_iql_roles(actor, phase, *, metadata_only, restoring):
    from sequifier.rl.models import critic_history_options, make_iql_models
    from sequifier.rl.phase_optimization import PhaseRoles

    history = critic_history_options(
        actor, phase, metadata_only=metadata_only, restoring=restoring
    )
    value, q1, q2, targets = make_iql_models(
        actor.execution_config, actor.interface_name, actor.config.actions, **history
    )
    return PhaseRoles(
        {
            "actor": actor.network,
            "value": value.eval(),
            "critic": TwinQ([q1, q2]).eval(),
        },
        {f"target_q{i}": target.eval() for i, target in enumerate(targets, 1)},
        {
            "value": ["head"],
            "q1": ["action_embedding", "head"],
            "q2": ["action_embedding", "head"],
        },
    )


def _select(inputs, mask):
    return type(inputs)(
        {name: gather_tensor(value, mask) for name, value in inputs.inputs.items()},
        {name: value[mask] for name, value in inputs.metadata.items()},
    )


def _polyak(target, source, rate):
    with torch.no_grad():
        parameters = canonicalize_state_dict(dict(source.named_parameters()))
        targets = canonicalize_state_dict(dict(target.named_parameters()))
        if targets.keys() != parameters.keys():
            raise ValueError("IQL online and target parameter names differ")
        for name, target_parameter in targets.items():
            parameter = parameters[name]
            if target_parameter.shape != parameter.shape:
                raise ValueError(f"Incompatible target parameter {name}")
            if isinstance(parameter, DTensor) != isinstance(target_parameter, DTensor):
                raise ValueError(f"Incompatible target shard {name}")
            if isinstance(parameter, DTensor) and (
                not isinstance(target_parameter, DTensor)
                or parameter.placements != target_parameter.placements
                or parameter.device_mesh != target_parameter.device_mesh
            ):
                raise ValueError(f"Incompatible target shard {name}")
            if isinstance(target_parameter, DTensor) and isinstance(parameter, DTensor):
                target_parameter.to_local().lerp_(parameter.to_local(), rate)
            else:
                target_parameter.lerp_(parameter, rate)
        buffers = canonicalize_state_dict(dict(source.named_buffers()))
        for name, target_buffer in canonicalize_state_dict(
            dict(target.named_buffers())
        ).items():
            target_buffer.copy_(buffers[name])


def _evaluate(
    actor,
    value,
    critic,
    dataset: IQLTransitionDataset,
    batch_size,
    gamma,
    device,
    strategy=None,
):
    totals = {"action_nll": 0.0, "bellman_residual": 0.0, "value": 0.0, "q_min": 0.0}
    with torch.no_grad():
        positions = (
            range(strategy.rank, len(dataset), strategy.world_size)
            if strategy
            else range(len(dataset))
        )
        for start in range(0, len(positions), batch_size):
            indices = positions[start : start + batch_size]
            with collated_windows((dataset[i] for i in indices), device) as batch:
                action = batch.action
                v = value(batch.current)
                q = torch.minimum(*critic(batch.current, action))
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
                del batch, action
    result = {key: value / len(dataset) for key, value in totals.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("non-finite IQL held-out likelihood or critic diagnostics")
    return result


def run_iql_phase(context: PhaseContext[IQLPhaseConfig, OfflineData]):
    actor, phase = context.actor.eval(), context.phase
    data = context.data.prepare_transitions()
    session = PhaseSession(context)
    value = session.calls["value"]
    critic = session.calls["critic"]
    q1, q2 = cast(TwinQ, session.modules["critic"])
    targets = session.calls["target_q1"], session.calls["target_q2"]
    if not 0 <= session.transaction <= phase.budget.iterations:
        raise ValueError("IQL checkpoint exceeds phase update budget")
    if any(steps != session.transaction for steps in session.steps.values()):
        raise ValueError("IQL optimizer clocks disagree with committed updates")

    def evaluate(**record):
        session.evaluate_validation(
            lambda: _evaluate(
                actor,
                value,
                critic,
                data[1],
                phase.evaluation.batch_size,
                phase.gamma,
                actor.device,
                session.strategy,
            ),
            **record,
        )

    session.evaluate_initial(evaluate)
    for update in range(session.transaction + 1, phase.budget.iterations + 1):
        if session.stopping.triggered:
            break
        indices = torch.randint(
            len(data[0]), (phase.batch_size,), generator=session.sampler
        )
        indices, scale = session.partition(indices)
        scale *= len(indices) / phase.batch_size
        with collated_windows(
            (data[0][int(i)] for i in indices), actor.device
        ) as batch:
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
                if session.strategy.sum(int(batch.bootstrap.any())):
                    if batch.bootstrap.any():
                        successor_value[batch.bootstrap] = value(
                            _select(batch.successor, batch.bootstrap)
                        )
                    else:
                        value(batch.current)
                target = batch.reward + phase.gamma * successor_value
            delta = qbar - value(batch.current)
            v_loss = (
                torch.where(delta < 0, 1 - phase.expectile, phase.expectile)
                * delta.square()
            ).mean()
            first, second = critic(batch.current, batch.action)
            q_loss = ((first - target).square() + (second - target).square()).mean()
            actor_loss = -(
                weight * actor.evaluate_actions(batch.current, batch.action)
            ).mean()
            losses = {"value": v_loss, "critic": q_loss, "actor": actor_loss}
            session.zero_grad()
            for role, loss in losses.items():
                if not torch.isfinite(loss):
                    raise ValueError(f"non-finite IQL {role} objective")
                (loss * scale).backward()
            del batch
        session.step()
        _polyak(session.networks["target_q1"], q1, phase.target_update_rate)
        _polyak(session.networks["target_q2"], q2, phase.target_update_rate)

        def global_mean(tensor):
            return (
                session.strategy.sum(float(tensor.mean()) * scale)
                / session.strategy.world_size
            )

        training_losses = {
            f"{role}_loss": global_mean(loss.detach()) for role, loss in losses.items()
        }
        session.complete_update(
            final=update == phase.budget.iterations,
            training={
                **training_losses,
                "advantage_mean": global_mean(advantage),
                "weight_mean": global_mean(weight),
                "weight_cap_fraction": global_mean(
                    (weight >= phase.max_advantage_weight).float()
                ),
            },
            evaluate=evaluate,
        )
    return session.finish(
        session.optimizer_clocks(),
        {"data": context.data.contract},
    )
