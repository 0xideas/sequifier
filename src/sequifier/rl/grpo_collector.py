"""Sequential same-start groups with independent action/environment RNG streams."""

import torch

from sequifier.io.actions import action_batch
from sequifier.rl.environment import require_group_start, validate_start_token
from sequifier.rl.group_collection import (  # noqa: F401
    GroupCollection,
    GroupCollectionError,
)
from sequifier.rl.history import _json_payload_bytes, frame_storage_bytes


@torch.no_grad()
def collect_groups(
    context,
    adapter,
    start_group,
    count,
    generation,
    env_steps,
    restart_stream=0,
    *,
    workers=None,
    storage=None,
):
    if workers is not None:
        from sequifier.rl.parallel_grpo_collector import collect_parallel_groups

        return collect_parallel_groups(
            context,
            adapter,
            workers,
            start_group,
            count,
            generation,
            env_steps,
            restart_stream,
            storage=storage,
        )
    actor, reference, phase = context.actor.eval(), context.reference, context.phase
    require_group_start(adapter)
    with GroupCollection(
        context, adapter, count, generation, env_steps, restart_stream, storage=storage
    ) as collection:
        groups = []
        for group_index in range(start_group, start_group + count):
            token = collection.start(group_index)
            trajectories = []
            for sample in range(phase.collection.group_size):
                state = collection.trajectory(group_index, sample, token)
                cursor = state.cursor
                cursor.reset(
                    adapter.reset_group(
                        validate_start_token(token),
                        collection.environment_seed(group_index, sample),
                    ),
                    f"group-{group_index}-sample-{sample}",
                )
                collection.validate_initial(state)
                result = None
                for _ in range(phase.collection.max_steps):
                    frame = cursor.history.current
                    collection.account(frame_storage_bytes(frame) + 1024)
                    decision = cursor.decide(compute_entropy=False)
                    collection.account(4 * _json_payload_bytes(decision.action))
                    reference_score = float(
                        reference.evaluate_actions(
                            cursor.inputs,
                            action_batch(
                                [decision.action], actor.config, device=actor.device
                            ),
                        )[0]
                    )
                    collection.executed += 1
                    result = cursor.step(decision)
                    state.append(
                        frame, decision, reference_score, result, phase.reward_scale
                    )
                    if result.terminated or result.truncated:
                        break
                if result is None:
                    raise ValueError("GRPO requires a positive trajectory horizon")
                trajectories.append(collection.seal_trajectory(state, result))
            groups.append(collection.seal_group(trajectories))
        return collection.finish(groups)
