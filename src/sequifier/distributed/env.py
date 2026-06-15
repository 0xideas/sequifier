import os
from datetime import timedelta

import torch
import torch.distributed as dist
from beartype import beartype


@beartype
def setup_distributed_env(
    rank: int, local_rank: int, world_size: int, backend: str = "nccl"
):
    """Initialize torch.distributed with env defaults."""
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")

    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    if not dist.is_initialized():
        timeout_sec = int(os.environ.get("NCCL_TIMEOUT", 1800))

        device_id = (
            torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None
        )

        dist.init_process_group(
            backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_sec),
            device_id=device_id,
        )
