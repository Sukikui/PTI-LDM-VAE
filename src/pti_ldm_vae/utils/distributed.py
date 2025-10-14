from datetime import timedelta
import os
import torch
import torch.distributed as dist
from typing import Tuple


def setup_ddp(rank: int, world_size: int) -> Tuple[dist, torch.device]:
    """
    Setup Distributed Data Parallel training.

    Args:
        rank: Rank of the current process
        world_size: Total number of processes

    Returns:
        Tuple of (dist module, device)
    """
    print(f"Running DDP diffusion example on rank {rank}/world_size {world_size}.")
    print(f"Initing to IP {os.environ['MASTER_ADDR']}")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=36000),
        rank=rank,
        world_size=world_size,
    )
    dist.barrier()
    device = torch.device(f"cuda:{rank}")
    return dist, device