"""Phase-local categorical behavior cloning, including return conditioning."""

import torch

from sequifier.config.policy_plan import PolicyImitationPhaseConfig
from sequifier.io.batch import collate_sequence_batches
from sequifier.io.policy_window import action_global_ids, action_target
from sequifier.rl.offline_data import OfflineData
from sequifier.rl.run_state import (
    PhaseContext,
    Selection,
    actor_state,
    apply_gradients,
    restore_actor_state,
    validate_permutation,
)


def action_nll(actor, batch):
    batch = batch.to(actor.device)
    labels = batch.targets[action_target(actor.config)][:, -1]
    ids = torch.tensor(action_global_ids(actor.config), device=actor.device)
    actions = (labels[:, None] == ids[None, :]).long().argmax(dim=-1)
    return -actor.evaluate_actions(batch.model_inputs, actions).float()


@torch.no_grad()
def evaluate_imitation(actor, dataset, batch_size):
    actor.eval()
    total = 0.0
    for start in range(0, len(dataset), batch_size):
        batch = collate_sequence_batches(
            [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        )
        total += float(action_nll(actor, batch).sum())
    return total / len(dataset)


def run_imitation_phase(context: PhaseContext[PolicyImitationPhaseConfig, OfflineData]):
    actor, phase, data = context.actor.eval(), context.phase, context.data
    optimizer, scheduler = context.optimization["actor"].build(actor)
    sampler = torch.Generator().manual_seed(context.run.seed + context.index)
    restore_actor_state(context, optimizer, scheduler, sampler)
    state = context.resume_state or {}
    selection = Selection(phase, phase.monitor, phase.direction, state.get("selection"))
    transaction = state.get("transaction", 0)
    epoch, position = state.get("epoch", 0), state.get("position", 0)
    order, metrics = state.get("order"), list(state.get("metrics", []))
    training = data.training
    size = phase.batch_size * phase.accumulation_steps
    per_epoch = (len(training) + size - 1) // size
    if (
        not isinstance(epoch, int)
        or not 0 <= epoch <= phase.epochs
        or transaction != epoch * per_epoch + (position + size - 1) // size
    ):
        raise ValueError("imitation checkpoint epoch/transaction clocks disagree")
    if order is not None:
        validate_permutation(order, len(training), position)
        if position >= len(training) or position % size:
            raise ValueError(
                "imitation checkpoint is not a complete accumulation boundary"
            )
    elif position != 0:
        raise ValueError("imitation checkpoint has a sampler position without an order")
    if epoch == phase.epochs and (order is not None or position):
        raise ValueError("completed imitation checkpoint has a live sampler order")
    while epoch < phase.epochs:
        if order is None:
            order = list(training.sampler(sampler))
        group = order[position : position + size]
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for start in range(0, len(group), phase.batch_size):
            indices = group[start : start + phase.batch_size]
            batch = collate_sequence_batches([training[i] for i in indices])
            loss = action_nll(actor, batch).sum() / len(group)
            if not torch.isfinite(loss):
                raise ValueError("non-finite imitation action likelihood")
            loss.backward()
            total += float(loss.detach())
        apply_gradients(
            actor, optimizer, scheduler, context.optimization["actor"].gradient_clip
        )
        transaction += 1
        position += len(group)
        if position == len(training):
            epoch += 1
            position, order = 0, None
        final = epoch == phase.epochs
        if transaction % phase.evaluation_interval == 0 or final:
            score = evaluate_imitation(actor, data.validation, phase.batch_size)
            selection.observe(actor, score, transaction)
            metrics.append(
                {
                    "transaction": transaction,
                    "epochs_completed": epoch,
                    "training_action_nll": total,
                    phase.monitor: score,
                }
            )
        if transaction % phase.checkpoint_interval == 0 or final:
            context.commit(
                actor_state(
                    context,
                    optimizer,
                    scheduler,
                    sampler,
                    selection,
                    transaction,
                    metrics,
                    epoch=epoch,
                    position=position,
                    order=order,
                )
            )
    return selection.finish(
        context,
        transaction,
        {"optimizer_steps": transaction, "epochs": epoch},
        metrics,
        {"data": data.contract},
    )
