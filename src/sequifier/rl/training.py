"""Phase-local categorical behavior cloning, including return conditioning."""

import torch

from sequifier.config.policy_plan import PolicyImitationPhaseConfig
from sequifier.io.batch import collate_sequence_batches
from sequifier.io.policy_dataset import imitation_example
from sequifier.rl.batch_order import BatchOrder
from sequifier.rl.batching import batches as minibatches
from sequifier.rl.offline_data import OfflineData
from sequifier.rl.run_state import PhaseContext, PhaseSession


def action_nll(actor, batch):
    batch = batch.to(actor.device)
    actions = {name: value[:, -1] for name, value in batch.targets.items()}
    return -actor.evaluate_actions(batch.model_inputs, actions).float()


@torch.no_grad()
def evaluate_imitation(actor, dataset, batch_size):
    actor.eval()
    total = 0.0
    for start in range(0, len(dataset), batch_size):
        batch = collate_sequence_batches(
            [
                imitation_example(dataset, i)
                for i in range(start, min(start + batch_size, len(dataset)))
            ]
        )
        total += float(action_nll(actor, batch).sum())
    return total / len(dataset)


def run_imitation_phase(context: PhaseContext[PolicyImitationPhaseConfig, OfflineData]):
    actor, phase, data = context.actor.eval(), context.phase, context.data
    session = PhaseSession(context)
    state = session.state
    training = data.training
    batches = BatchOrder(
        len(training),
        phase.batch_size * phase.accumulation_batches,
        context.run.seed + context.index,
        sample=training.sampler,
        state=state["batch_order"] if state else None,
    )
    if (
        batches.epoch > phase.epochs
        or session.transaction != batches.transactions
        or session.steps["actor"] != session.transaction
    ):
        raise ValueError("imitation checkpoint epoch/transaction clocks disagree")
    if batches.epoch == phase.epochs and batches.order is not None:
        raise ValueError("completed imitation checkpoint has a live sampler order")
    while batches.epoch < phase.epochs:
        group = batches.next()
        session.zero_grad()
        total = 0.0
        for indices in minibatches(group, phase.batch_size):
            batch = collate_sequence_batches(
                [imitation_example(training, i) for i in indices]
            )
            loss = action_nll(actor, batch).sum() / len(group)
            if not torch.isfinite(loss):
                raise ValueError("non-finite imitation action likelihood")
            loss.backward()
            total += float(loss.detach())
        session.step()
        session.transaction += 1
        final = batches.epoch == phase.epochs
        if session.should_evaluate(final):
            score = evaluate_imitation(actor, data.validation, phase.batch_size)
            session.record_evaluation(
                score,
                epochs_completed=batches.epoch,
                training_action_nll=total,
                **{phase.evaluation.monitor.metric: score},
            )
        session.checkpoint(final, batch_order=batches.state_dict())
    return session.finish(
        {"optimizer_steps": session.steps["actor"], "epochs": batches.epoch},
        {"data": data.contract},
    )
