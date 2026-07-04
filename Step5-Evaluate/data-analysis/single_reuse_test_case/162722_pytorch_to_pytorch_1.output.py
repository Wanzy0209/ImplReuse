import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def test_reduce(rank, world_size):
    setup(rank, world_size)

    # Adapted context: Using tensor shapes similar to the Transformer model in the bug report
    # (Batch size, Sequence length, Embedding size)
    batch_size = 2
    seq_length = 10
    embed_size = 256
    
    # Create a tensor on each process. 
    # We fill it with (rank + 1) to distinguish contributions from different processes.
    tensor = torch.ones(batch_size, seq_length, embed_size) * (rank + 1)

    # Original API: torch.compile (caused numerical inconsistency)
    # Similar API: torch.distributed.reduce (verify numerical consistency of reduction)
    
    # Perform the reduce operation: Sum all tensors to the destination process (rank 0)
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        # Calculate the expected value: Sum of 1 + 2 + ... + world_size
        expected_sum = sum(range(1, world_size + 1))
        expected_tensor = torch.ones(batch_size, seq_length, embed_size) * expected_sum

        # Verify numerical consistency
        # The original bug reported severe numerical inconsistencies. 
        # Here we assert that the distributed reduction maintains exact numerical consistency.
        assert torch.allclose(tensor, expected_tensor), \
            f"Numerical inconsistency detected in torch.distributed.reduce. Expected {expected_tensor[0,0,0]}, got {tensor[0,0,0]}"
        
        print("Test passed: torch.distributed.reduce maintained numerical consistency.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Spawn 2 processes to simulate a distributed environment
    mp.spawn(test_reduce, args=(world_size,), nprocs=world_size, join=True)