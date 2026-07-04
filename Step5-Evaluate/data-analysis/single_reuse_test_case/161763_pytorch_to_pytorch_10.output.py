import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def foo(x):
    # Original function from the bug report
    c = torch.tensor(7, dtype=torch.uint8)
    return c+x, torch.neg(c), torch.neg(c)+x

def run(rank, size):
    # Initialize the process group
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{os.environ['MASTER_PORT']}",
        rank=rank,
        world_size=size
    )

    # Create input data, varying seed by rank to ensure different data per process
    torch.manual_seed(0 + rank)
    x = torch.randn(2, 2, dtype=torch.float32)
    
    # Execute the function
    res = foo(x)

    # Prepare list for gathering
    if rank == 0:
        gathered_res = [None] * size
    else:
        gathered_res = None

    # Call the similar API: torch.distributed.gather_object
    # We gather the results from all ranks to rank 0
    dist.gather_object(res, gathered_res, dst=0)

    # Verify the results on the destination rank
    if rank == 0:
        for r in range(size):
            # Re-calculate expected result for rank r
            torch.manual_seed(0 + r)
            x_expected = torch.randn(2, 2, dtype=torch.float32)
            expected_res = foo(x_expected)
            
            gathered_tuple = gathered_res[r]
            
            # Assert that gathered objects match the expected computation
            # This verifies that gather_object correctly handles the uint8 and float tensors
            # involved in the original bug's logic.
            assert torch.equal(gathered_tuple[0], expected_res[0]), f"Rank {r} res[0] mismatch"
            assert torch.equal(gathered_tuple[1], expected_res[1]), f"Rank {r} res[1] mismatch"
            assert torch.equal(gathered_tuple[2], expected_res[2]), f"Rank {r} res[2] mismatch"
            
        print("Test passed: torch.distributed.gather_object correctly handled uint8 tensor computations.")

    dist.destroy_process_group()

if __name__ == "__main__":
    # Set a random port for the communication
    os.environ['MASTER_PORT'] = '29500'
    size = 2
    mp.spawn(run, args=(size,), nprocs=size, join=True)