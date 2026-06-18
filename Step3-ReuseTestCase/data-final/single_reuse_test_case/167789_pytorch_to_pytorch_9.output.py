import sys
import torch
import torch.distributed as dist
import os

# Minimal setup to allow torch.distributed API to run in a single-process environment
# This is necessary because torch.distributed.broadcast_object_list requires an initialized process group.
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
if not dist.is_initialized():
    dist.init_process_group(backend='gloo', rank=0, world_size=1)

def fn(obj_list, n):
    if n == 0:
        return obj_list
    # Recursive call
    res = fn(obj_list, n - 1)
    # Adaptation: Replace the original arithmetic logic with the similar API call
    # torch.distributed.broadcast_object_list broadcasts a list of objects to the group
    dist.broadcast_object_list(res, src=0)
    return res

@torch.compile(backend="eager")
def outer():
    # Create a list of tensors to broadcast
    data = [torch.ones(3)]
    return fn(data, 1000)

# The bug report indicates that sys.setrecursionlimit is ignored by torch.compile
# We set it high to test if the RecursionError still occurs
sys.setrecursionlimit(10000000)

try:
    outer()
    print("Test passed: No RecursionError encountered.")
except RecursionError as e:
    print(f"Test failed: RecursionError occurred despite high recursion limit. {e}")
finally:
    # Clean up distributed group
    if dist.is_initialized():
        dist.destroy_process_group()