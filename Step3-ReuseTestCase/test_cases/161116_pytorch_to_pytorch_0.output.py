import torch
import torch.distributed as dist
import os

def main():
    """
    Test case for initializing the NCCL process group.
    This script is designed to be run with torchrun or similar launchers
    on a multi-GPU system (specifically targeting the NVL72 issue context).
    """
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. This test requires CUDA.")
        return

    # Retrieve environment variables set by the launcher
    # Defaults are provided for single-process debugging, though the bug 
    # manifests in distributed scenarios.
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Set the device for the current process
    torch.cuda.set_device(local_rank)

    try:
        # Initialize the process group using the NCCL backend.
        # The 'device_id' argument is explicitly passed here, which is 
        # relevant to the reported issue on NVL72 clusters.
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            device_id=local_rank
        )

        # Synchronize all processes to ensure initialization completes globally
        dist.barrier()

        print(f"Process group initialized successfully on Rank {rank}/{world_size} (Local Rank: {local_rank})")

    except Exception as e:
        print(f"Error initializing process group on Rank {rank}: {e}")
        raise
    finally:
        # Clean up the process group
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()