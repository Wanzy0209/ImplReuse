import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# --- Fix for missing torch.compile (PyTorch < 2.0) ---
if not hasattr(torch, 'compile'):
    print("Warning: torch.compile not found. Using dummy decorator (eager execution).")
    def dummy_compile(*args, **kwargs):
        # Support both @torch.compile and @torch.compile(...)
        if args and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator
    torch.compile = dummy_compile

# --- Fix for missing torch._dynamo ---
if not hasattr(torch, '_dynamo'):
    class _DummyDynamo:
        @staticmethod
        def graph_break():
            pass
    torch._dynamo = _DummyDynamo()
# -----------------------------------------------------

def setup(rank, world_size):
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

@torch.compile(backend="eager")
def fn(obj, i, rank, world_size):
    # Replicate the bug scenario: conditional graph break
    if i == 1:
        torch._dynamo.graph_break()
    
    # Adaptation: Use torch.distributed.gather_object instead of tensor addition
    if rank == 0:
        gather_list = [None] * world_size
    else:
        gather_list = None
        
    dist.gather_object(obj, gather_list, dst=0)
    
    # Return result for verification
    if rank == 0:
        return gather_list
    return None

def run(rank, world_size):
    setup(rank, world_size)
    
    # Input object to gather
    obj = f"object_from_rank_{rank}"
    
    # Call 1: i=0 (Standard path)
    res1 = fn(obj, 0, rank, world_size)
    
    # Call 2: i=1 (Graph break path - triggers the bug scenario)
    res2 = fn(obj, 1, rank, world_size)
    
    # Call 3: i=2 (Standard path after recompilation)
    res3 = fn(obj, 2, rank, world_size)

    if rank == 0:
        expected = [f"object_from_rank_{r}" for r in range(world_size)]
        # Assertions to verify correctness and ensure the graph wasn't empty/malformed
        assert res1 == expected, f"Test failed on i=0: {res1} != {expected}"
        assert res2 == expected, f"Test failed on i=1 (Graph Break): {res2} != {expected}"
        assert res3 == expected, f"Test failed on i=2: {res3} != {expected}"
        print("Test passed successfully.")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Use spawn to launch processes for distributed testing
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)