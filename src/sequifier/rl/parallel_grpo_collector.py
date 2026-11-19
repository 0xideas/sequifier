"""Explicitly enabled synchronous GRPO collection across independent simulators."""

import math
from collections import deque

import torch

from sequifier.io.actions import action_record
from sequifier.rl.collection import PolicyDecision
from sequifier.rl.environment import validate_start_token
from sequifier.rl.group_collection import GroupCollection
from sequifier.rl.history import frame_storage_bytes
from sequifier.rl.simulator_workers import require_parallel_group_start
from sequifier.rl.window_storage import collated_windows


@torch.no_grad()
def _decide(actor, reference, cursors):
    with collated_windows(
        (cursor.inputs for cursor in cursors), actor.device, borrowed=True
    ) as inputs:
        distribution = actor.distribution(inputs)
        generators = [cursor.generator for cursor in cursors]
        # Decode and validate distributions across the full environment batch.
        # Only random draws use individual generators; no per-row decoder calls.
        selected = distribution.select_and_score(False, generators)
        scores = (
            torch.stack(
                [
                    selected.log_prob,
                    reference.evaluate_actions(inputs, selected.decision),
                ],
                dim=1,
            )
            .cpu()
            .tolist()
        )
        del inputs, distribution
    host_decision = selected.decision.cpu()
    host_actions = actor.action_plan.to_record(host_decision)
    decisions = []
    for index, (old, ref) in enumerate(scores):
        if not math.isfinite(old) or not math.isfinite(ref):
            raise ValueError("non-finite GRPO policy/reference action score")
        action = action_record(host_actions, index)
        decisions.append(
            (
                PolicyDecision(
                    actor.action_plan.domain_action(host_decision, index),
                    action,
                    old,
                    0.0,
                    0.0,
                ),
                ref,
            )
        )
    return decisions


@torch.no_grad()
def collect_parallel_groups(
    context,
    adapter,
    workers,
    start_group,
    count,
    generation,
    env_steps,
    restart_stream,
    *,
    storage=None,
):
    actor, reference, phase = (
        context.actor.eval(),
        context.reference.eval(),
        context.phase,
    )
    require_parallel_group_start(adapter)
    pending, active, completed = deque(), {}, {}
    with GroupCollection(
        context, adapter, count, generation, env_steps, restart_stream, storage=storage
    ) as collection:
        from sequifier.rl.history import _json_payload_bytes

        account = collection.account
        for group in range(start_group, start_group + count):
            token = collection.start(group)
            completed[group] = []
            pending.extend((group, sample, token) for sample in range(phase.group_size))
        while pending or active:
            resetting = []
            for slot in range(len(workers)):
                if slot in active or not pending:
                    continue
                group, sample, token = pending.popleft()
                active[slot] = collection.trajectory(
                    group, sample, token, deferred=True
                )
                workers.send(
                    slot,
                    "reset_group",
                    validate_start_token(token),
                    collection.environment_seed(group, sample),
                )
                resetting.append(slot)
            for slot in resetting:
                state = active[slot]
                with workers.response(slot) as response:
                    state.cursor.reset(
                        response, f"group-{state.group}-sample-{state.sample}"
                    )
                    del response
                    frame = state.cursor.history.current
                    collection.validate_initial(state)
                    # Include live history before any environment step is issued.
                    account(frame_storage_bytes(frame) + 1024)
                account(
                    sum(
                        value.numel() * value.element_size()
                        for value in (
                            *state.cursor.inputs.inputs.values(),
                            *state.cursor.inputs.metadata.values(),
                        )
                    ),
                    window=True,
                )
            decisions = {}
            slots = sorted(active)
            for start in range(0, len(slots), phase.rollout_inference_batch_size):
                chunk = slots[start : start + phase.rollout_inference_batch_size]
                scored = _decide(actor, reference, [active[s].cursor for s in chunk])
                decisions.update(zip(chunk, scored))
            for decision, _ in decisions.values():
                account(4 * _json_payload_bytes(decision.action))
            for slot, (decision, _) in decisions.items():
                # Count attempted actions conservatively before sending: a failed
                # send must never hide a potentially executed simulator action.
                collection.executed += 1
                workers.send(slot, "step", decision.domain_action)
            for slot, (decision, ref) in decisions.items():
                state = active[slot]
                frame = state.cursor.history.current
                with workers.response(slot) as response:
                    result = state.cursor.accept_step(decision, response)
                    del response
                    state.append(frame, decision, ref, result, phase.reward_scale)
                    # Include the successor even on the last step of a trajectory.
                    if result.observation is not None:
                        account(
                            frame_storage_bytes(state.cursor.history.current) + 1024
                        )
                if (
                    result.terminated
                    or result.truncated
                    or len(state.actions) == phase.max_steps
                ):
                    completed[state.group].append(
                        collection.seal_trajectory(state, result)
                    )
                    del active[slot]
        return collection.finish(
            [collection.seal_group(group) for group in completed.values()]
        )
