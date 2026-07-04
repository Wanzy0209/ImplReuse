import os
import torch
import torch.distributed as dist
import tempfile

def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """Verification that we have at least 2 gpus to run dist examples"""
    has_gpu = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    return has_gpu and gpu_count >= min_gpus

def main():
    _min_gpu_count = 2
    # Allow running with fewer GPUs for testing purposes, but warn
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        print(f"Warning: Unable to locate sufficient {_min_gpu_count} gpus. Running with available resources.")

    # Setup rank and device similar to the original example
    # Use RANK if available (global rank), otherwise LOCAL_RANK, otherwise 0
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    # Default to world_size 1 if not running in a distributed environment
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)
        print(f"Running on rank {rank} on device {device}")
    else:
        device = torch.device("cpu")
        print(f"Running on device {device}")

    # Original API Call: Initialize the process group
    if not dist.is_initialized():
        # Use FileStore to avoid dependency on environment variables (env://)
        # This allows the script to run standalone without torchrun
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "dist_store")
            store = dist.FileStore(store_path, world_size)
            
            # Determine backend
            if hasattr(dist, "get_default_backend_for_device"):
                backend = dist.get_default_backend_for_device(device)
                # Note: FileStore works with Gloo. If backend is NCCL, 
                # we fallback to Gloo for this standalone test context.
                if backend == "nccl":
                    backend = "gloo"
            else:
                backend = "gloo"
            
            dist.init_process_group(
                backend=backend, 
                store=store, 
                rank=rank, 
                world_size=world_size
            )

    # --- Test Case for Similar API: torch.distributed.get_process_group_ranks ---
    
    # Get the default process group (WORLD)
    process_group = dist.group.WORLD
    
    # Call the similar API
    ranks = dist.get_process_group_ranks(process_group)
    
    # Verify the results
    # Use the actual world size from the initialized process group
    current_world_size = dist.get_world_size()
    
    # Assertion 1: The return type should be a list
    assert isinstance(ranks, list), f"Expected list, got {type(ranks)}"
    
    # Assertion 2: The number of ranks should match the world size
    assert len(ranks) == current_world_size, f"Expected {current_world_size} ranks, got {len(ranks)}"
    
    # Assertion 3: The current rank should be present in the list
    assert rank in ranks, f"Current rank {rank} not found in group ranks {ranks}"
    
    # Assertion 4: Ranks should be ordered by group rank (global rank)
    assert ranks == sorted(ranks), f"Ranks {ranks} are not sorted"
    
    print(f"Rank {rank}: Test passed. Group ranks: {ranks}")

if __name__ == "__main__":
    main()