"""Phase-local categorical behavior cloning, including return conditioning."""

import torch

from sequifier.config.policy_plan import PolicyImitationPhaseConfig
from sequifier.io.batch import collate_sequence_batches
from sequifier.io.policy_window import action_global_ids, action_target
from sequifier.rl.batch_order import BatchOrder
from sequifier.rl.offline_data import OfflineData
from sequifier.rl.run_state import (
    PhaseContext,
    Selection,
    actor_state,
    apply_gradients,
    phase_events,
    restore_actor_state,
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
    restore_actor_state(context, optimizer, scheduler)
    state = context.resume_state or {}
    selection = Selection(phase, phase.monitor, phase.direction, state.get("selection"))
    transaction = state.get("transaction", 0)
    metrics = list(state.get("metrics", []))
    training = data.training
    batches = BatchOrder(
        len(training),
        phase.batch_size * phase.accumulation_steps,
        context.run.seed + context.index,
        sample=training.sampler,
        state=state["batch_order"] if state else None,
    )
    if batches.epoch > phase.epochs or transaction != batches.transactions:
        raise ValueError("imitation checkpoint epoch/transaction clocks disagree")
    if batches.epoch == phase.epochs and batches.order is not None:
        raise ValueError("completed imitation checkpoint has a live sampler order")
    while batches.epoch < phase.epochs:
        group = batches.next()
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
        evaluate, checkpoint = phase_events(
            phase, transaction, final=batches.epoch == phase.epochs
        )
        if evaluate:
            score = evaluate_imitation(actor, data.validation, phase.batch_size)
            selection.observe(actor, score, transaction)
            metrics.append(
                {
                    "transaction": transaction,
                    "epochs_completed": batches.epoch,
                    "training_action_nll": total,
                    phase.monitor: score,
                }
            )
        if checkpoint:
            context.commit(
                actor_state(
                    context,
                    optimizer,
                    scheduler,
                    selection,
                    transaction,
                    metrics,
                    batch_order=batches.state_dict(),
                )
            )
    return selection.finish(
        context,
        transaction,
        {"optimizer_steps": transaction, "epochs": batches.epoch},
        metrics,
        {"data": data.contract},
    )
