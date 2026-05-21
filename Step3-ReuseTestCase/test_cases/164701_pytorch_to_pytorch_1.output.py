import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

def setup():
    # Initialize the process group
    dist.init_process_group(
        backend="nccl",  # Using nccl to match the CUDA context of the original bug
        init_method="tcp://127.0.0.1:29500",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"])
    )
    torch.cuda.set_device(int(os.environ["RANK"]))

def cleanup():
    dist.destroy_process_group()

def reduce_state_eager(state, new_data, dev_null):
    """
    Eager version: Reduces new_data into state on rank 0.
    Mimics the structure of the original slide_to_the_left2.
    """
    # Perform the reduce operation: sum new_data from all ranks into rank 0's new_data buffer
    dist.reduce(new_data, dst=0)
    
    if dist.get_rank() == 0:
        # Update the state tensor with the reduced result
        state.copy_(new_data)
        # Update dev_null to mimic the original bug's pattern of multiple writes
        dev_null.copy_(new_data)

def run(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    # Setup distributed environment
    setup()

    device = torch.device(f"cuda:{rank}")
    
    # Initialize tensors
    # Rank 0 holds the state, initialized to zeros
    # All ranks hold new_data, initialized to rank value
    state = torch.zeros([4, 2048, 1024], device=device) if rank == 0 else None
    new_data = torch.ones([4, 2048, 1024], device=device) * (rank + 1)
    dev_null = torch.zeros([4, 2048, 1024], device=device)

    # 1. Run Eager Version
    # We need to reset inputs for the second run if we were running sequentially on the same rank,
    # but here we are comparing behavior across runs or just verifying correctness.
    # For this test, we will verify the eager version works as expected first.
    
    # Create copies for the compiled run to ensure fresh inputs
    state_compiled = torch.zeros([4, 2048, 1024], device=device) if rank == 0 else None
    new_data_compiled = new_data.clone()
    dev_null_compiled = torch.zeros([4, 2048, 1024], device=device)

    # Run eager
    reduce_state_eager(state, new_data, dev_null)
    
    # 2. Run Compiled Version
    # Compile the function
    reduce_state_compiled_fn = torch.compile(reduce_state_eager)
    
    # Run compiled
    reduce_state_compiled_fn(state_compiled, new_data_compiled, dev_null_compiled)

    # 3. Assertions
    if rank == 0:
        # Expected result: sum of (rank + 1) for rank 0 and rank 1
        # Rank 0 contributes 1, Rank 1 contributes 2. Sum = 3.
        expected_value = 3.0
        
        # Check eager result
        assert torch.all(state == expected_value), f"Eager mismatch! Expected {expected_value}, got {state.unique()}"
        
        # Check compiled result
        assert torch.all(state_compiled == expected_value), f"Compiled mismatch! Expected {expected_value}, got {state_compiled.unique()}"
        
        # Check that compiled matches eager
        assert torch.all(state == state_compiled), "Eager and Compiled results differ!"
        
        print(f"Rank {rank}: Test Passed. Eager and Compiled results match.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. This test requires CUDA to run.")
    elif torch.cuda.device_count() < world_size:
        print(f"This test requires at least {world_size} CUDA devices.")
    else:
        mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)