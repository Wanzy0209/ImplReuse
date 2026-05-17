import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use gloo backend for CPU compatibility
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def foo(x):
    # Adapted function using torch.distributed.reduce
    # We perform a reduction to rank 0
    dist.reduce(x, dst=0)
    return x

def run(rank, world_size):
    setup(rank, world_size)

    # Create input tensor
    # Use different values per rank to verify the reduction logic
    x = torch.ones(4, 4) * (rank + 1)

    # Eager execution
    eager_res = foo(x.clone())

    # Compiled execution
    cfoo = torch.compile(foo)
    compile_res = cfoo(x.clone())

    # Assertions
    if rank == 0:
        # Calculate expected sum: 1 + 2 + ... + world_size
        expected_sum = sum(range(1, world_size + 1))
        
        # Verify eager result
        assert torch.all(eager_res == expected_sum), "Eager result mismatch"
        
        # Verify compiled result
        assert torch.all(compile_res == expected_sum), "Compiled result mismatch"
        
        # Verify eager vs compiled
        torch.testing.assert_close(eager_res, compile_res)
        print("Test passed.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)