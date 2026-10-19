# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import sys
import time
from torch.distributed._functional_collectives import broadcast
from torch.distributed._tensor import DTensor, Replicate, DeviceMesh
torch._dynamo.config.capture_scalar_outputs = True
# Hard-coded master information
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"

def init_process():
    # Rank and World Size are still read from environment variables
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Use the hard-coded values for initialization
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{MASTER_ADDR}:{MASTER_PORT}',
        rank=rank,
        world_size=world_size
    )


    torch.cuda.set_device(rank)
    print(f"Initialized process group on rank {dist.get_rank()}, device {torch.cuda.current_device()}")

@torch.compile(dynamic=True)
def example_compile_with_cond(rank):
    """
    torch.cond version - most compile-friendly approach
    Replaces if-else with torch.cond for better compilation
    """
    rank = rank.item()
    # Using torch.cond for compile-friendly conditional execution
    # torch.cond requires a tensor predicate
    pred = torch.tensor(rank == 0)

    # NOTE: Cannot use requires_grad=True inside torch.cond lambdas
    tensor = torch.cond(
        pred,
        lambda: torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32 ,device="cuda"),
        lambda: torch.zeros(5, dtype=torch.float32, device="cuda")
    )
    return broadcast(tensor, src=0, group=dist.group.WORLD)


if __name__ == "__main__":
    try:
        init_process()
        rank = dist.get_rank()
        # Option 3: Functional collectives with torch.cond (most compile-friendly)
        tensor = example_compile_with_cond(torch.tensor([rank]))


    finally:
        # Clean up the process group to avoid resource leaks
        if dist.is_initialized():
            dist.destroy_process_group()