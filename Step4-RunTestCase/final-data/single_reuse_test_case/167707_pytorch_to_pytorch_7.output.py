import torch
import torch.distributed as dist
import os

# Initialize the distributed environment to allow calling get_global_rank
# Using a single process setup for a minimal runnable test
if not dist.is_initialized():
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='gloo', rank=0, world_size=1)

# Adapted test case for torch.distributed.get_global_rank
# Replaces the torch.profiler.profile logic from the original issue
group = dist.GroupMember.WORLD
group_rank = 0

# Call the similar API
global_rank = dist.get_global_rank(group, group_rank)

# Assertion to verify the API works correctly
assert global_rank == 0, f"Expected global rank 0, but got {global_rank}"

# Clean up
if dist.is_initialized():
    dist.destroy_process_group()