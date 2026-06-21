import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from contextlib import nullcontext

def run(rank, world_size):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Use 'gloo' backend as it is commonly available and supports object gathering
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Adapted from the original bug report structure:
    # 1. Profiler setup to monitor overhead
    # 2. Execution inside a DeviceContext
    # 3. Loop to stress the API
    
    # Conditionally add CUDA activity to avoid warnings if CUDA is not available
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        with_stack=True,
        activities=activities,
        on_trace_ready=lambda p: print(p.key_averages().table(sort_by="cuda_time_total", row_limit=5))
    ) as prof:
        # The original bug highlighted redundant H2D/D2H copies under this context
        # Fix: torch.device is not a context manager. Use torch.cuda.device for CUDA
        # and nullcontext for CPU.
        if torch.cuda.is_available():
            device_ctx = torch.cuda.device("cuda")
        else:
            device_ctx = nullcontext()

        with device_ctx:
            for i in range(10):
                # Create a non-tensor object to pass to the API
                # The original bug involved non-tensor arguments being handled inefficiently
                obj = {"iteration": i, "rank": rank, "data": list(range(5))}

                if rank == 0:
                    gather_list = [None] * world_size
                else:
                    gather_list = None

                # Call the similar API: torch.distributed.gather_object
                # This replaces the original 'mlp(x)' call site
                dist.gather_object(obj, gather_list, dst=0)

                # Basic verification on the destination rank
                if rank == 0:
                    assert gather_list is not None
                    assert len(gather_list) == world_size
                    for item in gather_list:
                        assert "iteration" in item
                        assert item["iteration"] == i

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Use multiprocessing to simulate a distributed environment
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)