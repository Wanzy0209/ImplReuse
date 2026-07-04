import torch
import torch.distributed as dist
import os

# Handle environments where torch.compile is not available (PyTorch < 2.0)
# This mocks the decorator to allow the test to run without crashing on AttributeError.
if not hasattr(torch, 'compile'):
    print("Warning: torch.compile not found. Mocking it as a pass-through function.")
    torch.compile = lambda func: func

def setup():
    # Initialize distributed environment for single-process testing
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if not dist.is_initialized():
        # Use 'gloo' backend for CPU compatibility in this test
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    setup()

    # The original bug involved a crash in torch.compile with int64 tensors (iota/arange).
    # We adapt the test to verify the similar API torch.distributed.broadcast_object_list
    # under torch.compile with similar data types.

    @torch.compile
    def f(obj_list):
        # Call the similar API
        dist.broadcast_object_list(obj_list, src=0)
        return obj_list

    # Create int64 data similar to the bug report (torch.ops.prims.iota is like arange)
    x = torch.arange(36, dtype=torch.int64)
    data = [x]

    # Run the compiled function
    out = f(data)

    # Verify the output matches the input
    assert len(out) == 1
    assert torch.equal(out[0], x)
    assert out[0].dtype == torch.int64

    cleanup()