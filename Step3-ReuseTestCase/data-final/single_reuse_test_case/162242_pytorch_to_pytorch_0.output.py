import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def test_distributed_scatter_determinism(rank, world_size):
    """
    Adapted test for torch.distributed.scatter.
    Note: torch.distributed.scatter does not take an 'index' argument like torch.scatter.
    It scatters data implicitly based on process rank. 
    This test verifies the determinism of the operation across multiple runs.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Setup data
    # Rank 0 holds the source data (scatter_list)
    # Other ranks hold None for scatter_list
    if rank == 0:
        # Mimicking 'src' from original test
        # Creating a list of tensors to scatter
        src_list = [torch.tensor([float(rank + i * world_size]) for i in range(2)]) 
        # Reshape to match the "batch" feel of the original test if needed, 
        # but scatter usually expects 1D per rank or matching shapes.
        # Let's use simple 1D tensors for clarity in distributed context.
        src_list = [torch.tensor([rank * 10.0 + i]) for i in range(world_size)]
    else:
        src_list = None

    # The tensor to receive data (mimicking 'inputs' from original test)
    inputs = torch.zeros(1)

    # Run multiple times to check for determinism (mimicking the 1000 iterations)
    # We reduce iterations to 10 for distributed testing speed, but logic is the same.
    num_iterations = 10
    for i in range(num_iterations):
        
        # Reset inputs to ensure we are receiving fresh data
        inputs.zero_()
        
        # The call to the similar API
        # torch.distributed.scatter(tensor, scatter_list=None, src=0, group=None)
        dist.scatter(inputs, scatter_list=src_list, src=0)

        # Verification
        # Rank i should receive src_list[i]
        expected_val = rank * 10.0 + rank
        expected_tensor = torch.tensor([expected_val])
        
        # Asserting the result matches the expected value
        # Using torch.allclose for float comparison
        if not torch.allclose(inputs, expected_tensor):
            print(f"Rank {rank}, Iteration {i}: Mismatch. Got {inputs}, Expected {expected_tensor}")
            raise AssertionError(f"Non-deterministic or incorrect behavior detected at Rank {rank}")

        # Synchronize to ensure all ranks finish this iteration before next
        dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(test_distributed_scatter_determinism, args=(world_size,), nprocs=world_size, join=True)