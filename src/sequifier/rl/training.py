"""Phase-local categorical behavior cloning, including return conditioning."""

import time

import torch

from sequifier.config.policy_plan import PolicyImitationPhaseConfig
from sequifier.io.policy_dataset import imitation_example
from sequifier.rl.batch_order import BatchOrder
from sequifier.rl.batching import batches as minibatches
from sequifier.rl.offline_data import OfflineData
from sequifier.rl.run_state import PhaseContext, PhaseSession
from sequifier.rl.window_storage import collated_windows


def action_nll(actor, batch):
    return _action_nll(actor, batch.to(actor.device))


def _action_nll(actor, batch):
    actions = {name: value[:, -1] for name, value in batch.targets.items()}
    return -actor.evaluate_actions(batch.model_inputs, actions).float()


@torch.no_grad()
def evaluate_imitation(actor, dataset, batch_size, strategy=None):
    actor.eval()
    total = 0.0
    indices = (
        range(strategy.rank, len(dataset), strategy.world_size)
        if strategy
        else range(len(dataset))
    )
    for indices in minibatches(indices, batch_size):
        with collated_windows(
            (imitation_example(dataset, i) for i in indices), actor.device
        ) as batch:
            total += float(_action_nll(actor, batch).sum())
            del batch
    return total / len(dataset)


def run_imitation_phase(context: PhaseContext[PolicyImitationPhaseConfig, OfflineData]):
    actor, phase, data = context.actor.eval(), context.phase, context.data
    session = PhaseSession(context)
    state = session.state
    training = data.training
    batches = BatchOrder(
        len(training),
        phase.batch_size * phase.accumulation_batches,
        context.identity.sampling_seed,
        sample=training.sampler,
        state=state["batch_order"] if state else None,
    )
    if (
        batches.epoch > phase.budget.epochs
        or session.transaction != batches.transactions
        or session.steps["actor"] != session.transaction
    ):
        raise ValueError("imitation checkpoint epoch/transaction clocks disagree")
    if batches.epoch == phase.budget.epochs and batches.order is not None:
        raise ValueError("completed imitation checkpoint has a live sampler order")

    def evaluate(**record):
        started = time.monotonic()
        with session.rounds.scope():
            score = evaluate_imitation(
                actor, data.validation, phase.evaluation.batch_size, session.strategy
            )
        score = session.strategy.sum(score)
        session.record_evaluation(
            score,
            elapsed_seconds=time.monotonic() - started,
            epochs_completed=batches.epoch,
            **record,
            **{phase.evaluation.monitor.metric: score},
        )

    session.epoch = batches.epoch
    session.evaluate_initial(evaluate)
    while batches.epoch < phase.budget.epochs and not session.stopping.triggered:
        session.epoch = batches.epoch + 1
        position = batches.position
        group = batches.next()
        session.zero_grad()
        total = 0.0
        for microbatch in minibatches(group, phase.batch_size):
            indices, scale = session.partition(microbatch, position)
            position += len(microbatch)
            with collated_windows(
                (imitation_example(training, i) for i in indices), actor.device
            ) as batch:
                loss = _action_nll(actor, batch).sum() * scale / len(group)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite imitation action likelihood")
                session.optimization["actor"].backward(loss)
                total += float(loss.detach())
                del batch
        session.step()
        session.complete_update(
            final=batches.epoch == phase.budget.epochs,
            training={
                "training_action_nll": session.strategy.sum(total)
                / session.strategy.world_size
            },
            evaluate=evaluate,
            checkpoint=lambda: {"batch_order": batches.state_dict()},
        )
    return session.finish(
        {"optimizer_steps": session.steps["actor"], "epochs": batches.epoch},
        {"data": data.contract},
    )
