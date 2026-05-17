import torch
import torch.distributed as dist
import os
import random

def setup_distributed():
    """Initialize the process group for single-process testing."""
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        # Initialize with a single rank (world_size=1) for standalone testing
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

def cleanup_distributed():
    """Clean up the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def foo(obj_list):
    """
    Function using the similar API: torch.distributed.broadcast_object_list.
    This replaces the math operations from the original bug report.
    """
    # Broadcast the object list from source rank 0
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list

if __name__ == "__main__":
    setup_distributed()

    # Retain the configuration from the original bug report
    torch._dynamo.config.capture_scalar_outputs = True

    # Generate random input data similar to the original np.random setup
    random.seed(0)
    # Create a list of mixed objects to broadcast
    input_data = [random.randint(0, 100), "test_string", {"val": random.random()}]

    # Compile the function
    cfoo = torch.compile(foo)

    # Execute eager version
    eager_input = input_data.copy()
    eager_res = foo(eager_input)

    # Execute compiled version
    compiled_input = input_data.copy()
    compiled_res = cfoo(compiled_input)

    # Verify results match
    # Since broadcast_object_list modifies in-place, we compare the resulting lists
    try:
        assert eager_res == compiled_res, f"Mismatch: Eager {eager_res} vs Compiled {compiled_res}"
        print("Test passed: Eager and Compiled results match.")
    except AssertionError as e:
        print(f"Test failed: {e}")
    
    cleanup_distributed()