"""Canonical sigmoid DPO with bounded differentiable sequence scoring."""

import torch

from sequifier.config.policy_plan import DPOPhaseConfig
from sequifier.io.preference_dataset import (
    PreferenceData,
    dpo_loss,
    sequence_log_probabilities,
    sequence_score,
)
from sequifier.rl.run_state import (
    PhaseContext,
    Selection,
    actor_state,
    apply_gradients,
    make_optimization,
    restore_actor_state,
    validate_permutation,
)


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
    optimizer, scheduler = make_optimization(actor, phase)
    sampler = torch.Generator().manual_seed(context.run.seed + context.index)
    restore_actor_state(context, optimizer, scheduler, sampler)
    state = context.resume_state or {}
    transaction, metrics = state.get("transaction", 0), list(state.get("metrics", []))
    if not 0 <= transaction <= phase.updates:
        raise ValueError("DPO checkpoint exceeds authored update budget")
    selection = Selection(phase, phase.monitor, phase.direction, state.get("selection"))
    order, position = state.get("order", []), state.get("position", 0)
    if state:
        validate_permutation(order, len(data.training), position)
        size = phase.batch_size * phase.accumulation_steps
        cycle = (len(data.training) + size - 1) // size
        expected = min(((transaction - 1) % cycle + 1) * size, len(data.training))
        if position != expected:
            raise ValueError(
                "DPO sampler position disagrees with optimizer transaction"
            )
    for update in range(transaction + 1, phase.updates + 1):
        if position == len(order):
            order, position = (
                torch.randperm(len(data.training), generator=sampler).tolist(),
                0,
            )
        group = order[position : position + phase.batch_size * phase.accumulation_steps]
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
        apply_gradients(actor, optimizer, scheduler, phase.gradient_clip)
        transaction, position = update, position + len(group)
        final = update == phase.updates
        if update % phase.evaluation_interval == 0 or final:
            held_out = evaluate_dpo(actor, reference, data.validation, phase)
            selection.observe(actor, held_out["dpo_loss"], update)
            metrics.append(
                {
                    "transaction": update,
                    "training_dpo_loss": loss_total,
                    **{f"validation_{key}": value for key, value in held_out.items()},
                }
            )
        if update % phase.checkpoint_interval == 0 or final:
            context.commit(
                actor_state(
                    context,
                    optimizer,
                    scheduler,
                    sampler,
                    selection,
                    update,
                    metrics,
                    order=order,
                    position=position,
                )
            )
    return selection.finish(
        context,
        transaction,
        {"optimizer_steps": transaction},
        metrics,
        {"data": data.contract, "reference_weight_hash": context.reference_identity},
    )
