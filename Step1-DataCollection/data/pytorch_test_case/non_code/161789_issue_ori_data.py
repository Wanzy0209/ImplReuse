import os
import torch
import rmm
from rmm.allocators.torch import rmm_torch_allocator

def bug(target_device):
    SIZE = 1024
    stream = torch.cuda.Stream(device=target_device)
    with torch.cuda.stream(stream):
        tensor = torch.stack([
                         torch.arange(SIZE, dtype=torch.int64, device=target_device),
                         torch.arange(SIZE, dtype=torch.int64, device=target_device)], dim=1)
        # tensor = torch.ones((SIZE,2),device=target_device) # same problem
        stream.synchronize()
        for i in range(100):
            # groups_unique = torch.unique_consecutive(tensor, dim=0) # this works
            print(f"{i}",end=",")
            groups_unique, group_idx = torch.unique_consecutive(tensor, return_inverse=True, dim=0)
            # produces RuntimeError: inclusive_scan failed to synchronize: cudaErrorIllegalAddress: an illegal memory access was encountered 
    stream.synchronize()

if __name__ == "__main__":
    if os.environ.get("USE_RMM", "False") == "True":
        rmm.reinitialize(pool_allocator=True, initial_pool_size = "4GiB")
        torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
        print("With RMM")

    target_device = "cuda:0"
    bug(target_device)