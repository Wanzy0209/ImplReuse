import torch
import torch.distributed as dist
import os

def main():
    # Fix: Handle missing environment variables for standalone execution
    # If the script is not run via torchrun/mp.spawn, these variables might not be set.
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

    # Setup from the original bug report
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    # Initialize the process group.
    # Note: On the specific NVL72 hardware mentioned in the bug report (>10 trays),
    # this call is expected to segfault due to the NCCL topology search issue.
    dist.init_process_group(backend='nccl', device_id=gpu_id)

    # Test the similar API: torch.distributed.get_process_group_ranks
    # This API retrieves the list of global ranks for the given process group.
    # We use the default WORLD group here.
    group = dist.group.WORLD
    ranks = dist.get_process_group_ranks(group)

    # Verify the result
    world_size = dist.get_world_size()
    expected_ranks = list(range(world_size))
    
    assert ranks == expected_ranks, f"Expected ranks {expected_ranks}, but got {ranks}"
    
    # Synchronize all processes
    dist.barrier()

if __name__ == "__main__":
    main()