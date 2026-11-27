"""Canonical sigmoid DPO with bounded differentiable sequence scoring."""

import torch
from torch.nn import functional as F

from sequifier.config.policy_plan import DPOPhaseConfig
from sequifier.io.preference_dataset import PreferenceData
from sequifier.rl.batch_order import BatchOrder
from sequifier.rl.batching import batches as minibatches
from sequifier.rl.run_state import PhaseContext, PhaseSession
from sequifier.rl.scoring import sequence_log_probabilities, sequence_score


def dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta):
    margin = (chosen.float() - rejected.float()) - (
        reference_chosen.float() - reference_rejected.float()
    )
    loss = -F.logsigmoid(beta * margin)
    if not torch.isfinite(margin).all() or not torch.isfinite(loss).all():
        raise ValueError("non-finite DPO margin/loss")
    return loss, margin


def pair_scores(actor, reference, dataset, pair, chunk_size):
    scores = []
    for policy in (actor, reference):
        for span in (pair.chosen, pair.rejected):
            scores.append(
                sequence_score(
                    policy, dataset.episode(span), span.start, span.end, chunk_size
                )
            )
    return scores[0], scores[1], scores[2], scores[3]


@torch.no_grad()
def evaluate_dpo(actor, reference, dataset, phase, strategy=None):
    totals = {
        key: 0.0
        for key in (
            "dpo_loss",
            "margin",
            "preference_accuracy",
            "chosen_length",
            "rejected_length",
        )
    }
    positions = (
        range(strategy.rank, len(dataset), strategy.world_size)
        if strategy
        else range(len(dataset))
    )
    for index in positions:
        pair = dataset.pairs[index]
        loss, margin = dpo_loss(
            *pair_scores(actor, reference, dataset, pair, phase.scoring_chunk_size),
            phase.beta,
        )
        totals["dpo_loss"] += float(loss)
        totals["margin"] += float(margin)
        totals["preference_accuracy"] += float(margin > 0)
        totals["chosen_length"] += pair.chosen.end - pair.chosen.start
        totals["rejected_length"] += pair.rejected.end - pair.rejected.start
    return {key: value / len(dataset) for key, value in totals.items()}


def run_dpo_phase(context: PhaseContext[DPOPhaseConfig, PreferenceData]):
    actor, phase, data, reference = (
        context.actor.eval(),
        context.phase,
        context.data,
        context.reference,
    )
    if reference is None:
        raise ValueError("DPO requires a frozen reference actor")
    session = PhaseSession(context)
    state = session.state
    if not 0 <= session.transaction <= phase.budget.iterations:
        raise ValueError("DPO checkpoint exceeds authored update budget")
    batches = BatchOrder(
        len(data.training),
        phase.batch_size * phase.accumulation_batches,
        context.identity.sampling_seed,
        state=state["batch_order"] if state else None,
    )
    if (
        session.transaction != batches.transactions
        or session.steps["actor"] != session.transaction
    ):
        raise ValueError("DPO sampler position disagrees with optimizer transaction")

    def evaluate(**record):
        session.evaluate_validation(
            lambda: evaluate_dpo(
                actor, reference, data.validation, phase, session.strategy
            ),
            **record,
        )

    session.evaluate_initial(evaluate)
    for update in range(session.transaction + 1, phase.budget.iterations + 1):
        if session.stopping.triggered:
            break
        position = batches.position
        group = batches.next()
        session.zero_grad()
        loss_total = 0.0
        width = session.strategy.world_size
        chunk_size = phase.scoring_chunk_size
        for indices in minibatches(group, width):
            rank = (session.strategy.rank - position) % width
            position += len(indices)
            pairs = [data.training.pairs[index] for index in indices]
            pair = pairs[rank] if rank < len(pairs) else pairs[0]
            scale = width / len(group) if rank < len(pairs) else 0.0
            spans = (pair.chosen, pair.rejected)
            rounds = [
                max(
                    (span.end - span.start + chunk_size - 1) // chunk_size
                    for span in alternatives
                )
                for alternatives in (
                    [p.chosen for p in pairs],
                    [p.rejected for p in pairs],
                )
            ]

            def chunks(policy, side):
                span = spans[side]
                return sequence_log_probabilities(
                    policy,
                    data.training.episode(span),
                    span.start,
                    span.end,
                    chunk_size,
                    rounds=rounds[side],
                )

            with torch.no_grad():
                scores = [
                    torch.stack([chunk.sum() for chunk in chunks(policy, side)]).sum()
                    for policy in (actor, reference)
                    for side in range(2)
                ]
                chosen, rejected, ref_chosen, ref_rejected = scores
                loss, margin = dpo_loss(
                    chosen, rejected, ref_chosen, ref_rejected, phase.beta
                )
                coefficient = -phase.beta * torch.sigmoid(-phase.beta * margin) * scale
                loss_total += float(loss) * scale
            for side, sign in enumerate((1, -1)):
                for chunk in chunks(actor, side):
                    session.optimization["actor"].backward(
                        sign * coefficient * chunk.sum()
                    )
        session.step()
        session.complete_update(
            final=update == phase.budget.iterations,
            training={"training_dpo_loss": session.strategy.sum(loss_total) / width},
            evaluate=evaluate,
            checkpoint=lambda: {"batch_order": batches.state_dict()},
        )
    return session.finish(
        {"optimizer_steps": session.steps["actor"]},
        {"data": data.contract, "reference_weight_hash": context.reference_identity},
    )
