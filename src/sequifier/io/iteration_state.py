from collections.abc import Sequence

import torch


def shared_int(value: int = 0) -> torch.Tensor:
    """Return a shared scalar int tensor visible to DataLoader worker copies."""
    state = torch.empty((), dtype=torch.int64)
    state.fill_(int(value))
    state.share_memory_()
    return state


def read_shared_int(state: torch.Tensor) -> int:
    return int(state.item())


def write_shared_int(state: torch.Tensor, value: int) -> None:
    state.fill_(int(value))


def resolve_resume_worker(
    start_batch: int,
    worker_id: int,
    num_workers: int,
    worker_batch_counts: Sequence[int],
) -> tuple[int, int]:
    """Map a physical worker to the logical worker/batch offset after resume.

    PyTorch requests IterableDataset samples from workers in worker-id order. To
    resume at a non-zero global batch without yielding skipped batches, worker 0
    is rotated onto the worker that would have produced start_batch in the
    uninterrupted iterator. Each logical worker then skips only the batches it
    owned before start_batch.
    """
    if num_workers <= 0:
        raise ValueError("num_workers must be positive.")
    if len(worker_batch_counts) != num_workers:
        raise ValueError("worker_batch_counts must match num_workers.")

    counts = [max(0, int(count)) for count in worker_batch_counts]
    total_batches = sum(counts)
    capped_start = min(max(0, int(start_batch)), total_batches)
    if capped_start == 0 or total_batches == 0:
        return worker_id, 0

    skips = [0] * num_workers
    active_workers = [i for i, count in enumerate(counts) if count > 0]
    consumed = 0
    start_worker = active_workers[0] if active_workers else 0

    while active_workers and consumed < capped_start:
        rounds_until_next_exhaustion = min(counts[i] - skips[i] for i in active_workers)
        block_batches = rounds_until_next_exhaustion * len(active_workers)
        remaining = capped_start - consumed

        if remaining >= block_batches:
            for i in active_workers:
                skips[i] += rounds_until_next_exhaustion
            consumed += block_batches
            active_workers = [i for i in active_workers if skips[i] < counts[i]]
            start_worker = active_workers[0] if active_workers else 0
            continue

        full_rounds, offset = divmod(remaining, len(active_workers))
        for i in active_workers:
            skips[i] += full_rounds
        for i in active_workers[:offset]:
            skips[i] += 1
        start_worker = active_workers[offset]
        consumed = capped_start

    logical_worker_id = (worker_id + start_worker) % num_workers
    return logical_worker_id, skips[logical_worker_id]


def skip_samples_for_batches(
    skip_batches: int, batch_size: int, total_samples: int
) -> int:
    return min(max(0, int(skip_batches)) * int(batch_size), max(0, int(total_samples)))
