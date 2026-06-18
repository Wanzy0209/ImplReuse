import torch
import torch.distributed as dist
import os
from typing import NamedTuple

def setup_distributed():
    """Initialize a minimal distributed environment for testing."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    if not dist.is_initialized():
        # Use 'gloo' backend for CPU compatibility in testing
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

def cleanup_distributed():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

class MyNamedTuple(NamedTuple):
    first: torch.Tensor
    second: torch.Tensor

class MyNamedTupleSubclass(MyNamedTuple):
    pass

def fn(tup: MyNamedTuple) -> MyNamedTuple:
    # Adaptation: Use torch.distributed.reduce instead of torch.compile logic
    # We reduce the 'first' tensor to rank 0 (which is this process)
    dist.reduce(tup.first, dst=0)
    
    extra_info = torch.tensor(4.0)
    tup.extra_info = extra_info  # Add dynamic attribute
    return tup

if __name__ == "__main__":
    setup_distributed()

    print("\nTesting NamedTuple with torch.distributed.reduce:")
    extended_tup = MyNamedTupleSubclass(first=torch.tensor([2.0]), second=torch.tensor(1.0))
    
    # Call the function containing the distributed API
    reduce_result = fn(extended_tup)
    
    # Verify if the dynamic attribute persists after the API call
    try:
        print(f"NamedTuple result: {reduce_result.extra_info}")
        assert reduce_result.extra_info == 4.0, "Attribute value mismatch"
        print("Test Passed: Dynamic attribute persisted.")
    except AttributeError as e:
        print(f"Test Failed: {e}")

    cleanup_distributed()