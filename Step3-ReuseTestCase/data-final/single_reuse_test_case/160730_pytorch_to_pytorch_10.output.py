import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_test(rank, world_size):
    setup(rank, world_size)

    # Define the function using the similar API (torch.distributed.gather_object)
    def foo(obj):
        # Adaptation: Use gather_object instead of torch.tan/torch.sin
        if rank == 0:
            gather_list = [None] * world_size
        else:
            gather_list = None
        
        dist.gather_object(obj, gather_list, dst=0)
        return gather_list

    # Compile the function
    cfoo = torch.compile(foo)

    # Create test object
    test_obj = f"rank_{rank}_data"

    # Run eager
    eager_res = foo(test_obj)
    dist.barrier()

    # Run compiled
    compile_res = cfoo(test_obj)
    dist.barrier()

    # Verify results (only rank 0 has the gathered list)
    if rank == 0:
        # Check if the compiled result matches the eager result
        # This mirrors the original test's assertion logic
        assert eager_res == compile_res, f"Eager and Compiled results differ: {eager_res} vs {compile_res}"
        print("Assertion passed: Eager and Compiled results match.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)