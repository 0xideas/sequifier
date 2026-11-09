"""Canonical sigmoid DPO with bounded differentiable sequence scoring."""

import torch
from torch.nn import functional as F

from sequifier.config.policy_plan import DPOPhaseConfig
from sequifier.io.preference_dataset import PreferenceData
from sequifier.rl.batch_order import BatchOrder
from sequifier.rl.run_state import (
    PhaseContext,
    Selection,
    actor_state,
    apply_gradients,
    phase_events,
    restore_actor_state,
)
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
def evaluate_dpo(actor, reference, dataset, phase):
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
    for pair in dataset.pairs:
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
    optimizer, scheduler = context.optimization["actor"].build(actor)
    restore_actor_state(context, optimizer, scheduler)
    state = context.resume_state or {}
    transaction, metrics = state.get("transaction", 0), list(state.get("metrics", []))
    if not 0 <= transaction <= phase.updates:
        raise ValueError("DPO checkpoint exceeds authored update budget")
    selection = Selection(phase, phase.monitor, phase.direction, state.get("selection"))
    batches = BatchOrder(
        len(data.training),
        phase.batch_size * phase.accumulation_steps,
        context.run.seed + context.index,
        state=state["batch_order"] if state else None,
    )
    if transaction != batches.transactions:
        raise ValueError("DPO sampler position disagrees with optimizer transaction")
    for update in range(transaction + 1, phase.updates + 1):
        group = batches.next()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for index in group:
            pair = data.training.pairs[index]
            # The derivative of the scalar DPO margin is known. First score without
            # graphs, then backpropagate each causal chunk with that exact derivative.
            # This bounds activation memory even for arbitrarily long alternatives.
            with torch.no_grad():
                scores = pair_scores(
                    actor, reference, data.training, pair, phase.scoring_chunk_size
                )
                loss, margin = dpo_loss(*scores, phase.beta)
                coefficient = (
                    -phase.beta * torch.sigmoid(-phase.beta * margin) / len(group)
                )
                loss_total += float(loss) / len(group)
            for span, sign in ((pair.chosen, 1), (pair.rejected, -1)):
                episode = data.training.episode(span)
                for chunk in sequence_log_probabilities(
                    actor, episode, span.start, span.end, phase.scoring_chunk_size
                ):
                    (sign * coefficient * chunk.sum()).backward()
        apply_gradients(
            actor, optimizer, scheduler, context.optimization["actor"].gradient_clip
        )
        transaction = update
        evaluate, checkpoint = phase_events(
            phase, update, final=update == phase.updates
        )
        if evaluate:
            held_out = evaluate_dpo(actor, reference, data.validation, phase)
            selection.observe(actor, held_out["dpo_loss"], update)
            metrics.append(
                {
                    "transaction": update,
                    "training_dpo_loss": loss_total,
                    **{f"validation_{key}": value for key, value in held_out.items()},
                }
            )
        if checkpoint:
            context.commit(
                actor_state(
                    context,
                    optimizer,
                    scheduler,
                    selection,
                    update,
                    metrics,
                    batch_order=batches.state_dict(),
                )
            )
    return selection.finish(
        context,
        transaction,
        {"optimizer_steps": transaction},
        metrics,
        {"data": data.contract, "reference_weight_hash": context.reference_identity},
    )
