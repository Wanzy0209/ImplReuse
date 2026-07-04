import torch
import torch.distributed as dist
import os
import sys

def test_get_group_rank():
    # Setup environment for single-process distributed run
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    # Test Case 1: Verify identity mapping for default group (WORLD)
    # Based on extracted info: "calling this function on the default process group returns identity"
    group = dist.group.WORLD
    global_rank = 0
    result_rank = dist.get_group_rank(group, global_rank)
    assert result_rank == global_rank, f"Expected {global_rank}, got {result_rank}"

    # Test Case 2: Verify handling for invalid global rank
    # PyTorch's get_group_rank returns -1 if the global rank is not part of the group,
    # rather than raising a ValueError.
    result_rank = dist.get_group_rank(group, 999)
    assert result_rank == -1, f"Expected -1 for rank not in group, got {result_rank}"

    # Cleanup
    dist.destroy_process_group()
    print("Test passed.")

if __name__ == "__main__":
    test_get_group_rank()