"""
Run:

torchrun --nproc_per_node=2 test_reduce.py
OR
python test_reduce.py (runs in single-process mode if env vars are missing)
"""

import torch
import torch.distributed as dist
import os

def main():
    # Setup similar to the bug report environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Ensure CUDA is available before setting device
    if not torch.cuda.is_available():
        print("CUDA is not available. This test requires CUDA.")
        return

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    
    # Initialize process group
    # Handle the case where the script is run without torchrun (missing RANK env var)
    if "RANK" not in os.environ:
        print("RANK environment variable not found. Initializing single-process distributed environment.")
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://localhost:29500",
            rank=0,
            world_size=1
        )
    else:
        dist.init_process_group(backend="nccl")
        
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Test Case 1: Basic torch.distributed.reduce functionality
    # Replaces the complex pipeline schedule with a direct test of the collective op.
    print(f"[Rank {rank}] Testing basic torch.distributed.reduce...")
    
    tensor = torch.ones(4, 4, device=device) * rank
    # Reduce sum to rank 0
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        # Sum of ranks 0 to N-1
        expected_sum = sum(range(world_size))
        expected = torch.ones(4, 4, device=device) * expected_sum
        assert torch.all(tensor == expected), f"Basic Reduce failed: {tensor} != {expected}"
        print(f"[Rank {rank}] Basic torch.distributed.reduce passed.")

    # Test Case 2: torch.distributed.reduce inside a torch.compiled function
    # This addresses the context of the original bug (compatibility with torch.compile)
    print(f"[Rank {rank}] Testing torch.distributed.reduce inside torch.compile...")
    
    @torch.compile
    def compiled_reduce_step(input_tensor):
        # Simulating a reduction step (e.g., gradient synchronization) within a compiled graph
        dist.reduce(input_tensor, dst=0, op=dist.ReduceOp.SUM)
        return input_tensor

    tensor_compiled = torch.ones(4, 4, device=device) * (rank + 10)
    
    # Execute the compiled function containing the reduce call
    compiled_reduce_step(tensor_compiled)
    
    if rank == 0:
        # Sum of (rank + 10) for rank 0 and 1 -> 10 + 11 = 21
        expected_sum_c = sum(r + 10 for r in range(world_size))
        expected_c = torch.ones(4, 4, device=device) * expected_sum_c
        assert torch.all(tensor_compiled == expected_c), f"Compiled Reduce failed: {tensor_compiled} != {expected_c}"
        print(f"[Rank {rank}] Compiled torch.distributed.reduce passed.")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()