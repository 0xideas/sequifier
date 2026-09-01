"""Compact distributed coordination used by source scheduling."""

import torch
import torch.distributed as dist


def broadcast_source_selection(
    selected_on_rank_zero: tuple[int, int, int, int, int] | None,
) -> tuple[int, int, int, int, int]:
    backend = dist.get_backend()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if backend == "nccl"
        else torch.device("cpu")
    )
    if dist.get_rank() == 0:
        if selected_on_rank_zero is None:
            raise ValueError("Rank zero must provide a source selection.")
        record = torch.tensor(selected_on_rank_zero, dtype=torch.int64, device=device)
    else:
        record = torch.empty(5, dtype=torch.int64, device=device)
    dist.broadcast(record, src=0)
    return tuple(int(value) for value in record.cpu().tolist())  # type: ignore[return-value]
