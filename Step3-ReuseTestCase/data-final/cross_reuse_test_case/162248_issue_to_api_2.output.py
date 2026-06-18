import torch
import torch.distributed as dist
import os
import multiprocessing as mp
import pytest

def is_distributed_available():
    """
    Helper function to check if distributed support is available.
    This mirrors the pattern of torch.backends.nnpack.is_available().
    """
    return torch.distributed.is_available()

def run_gloo_all_to_all(local_rank):
    """
    Worker function to test all_to_all with GLOO backend.
    Preserves the logic from the original bug report.
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = '2'

    try:
        # Initialize process group with GLOO backend
        group = dist.init_process_group(backend='gloo', rank=local_rank)
        
        world_size = 2
        output = list(torch.empty([world_size], dtype=torch.int64).chunk(world_size))
        input = list((torch.arange(world_size) + local_rank * world_size).chunk(world_size))
        
        # Attempt the operation that is documented but not supported
        dist.all_to_all(output, input, group=group)
        
        # If we reach here, the operation is supported (docs would be correct)
        dist.destroy_process_group()
        return "SUPPORTED"
        
    except RuntimeError as e:
        # Expected behavior based on the bug report
        if "Backend gloo does not support alltoall" in str(e):
            return "NOT_SUPPORTED"
        raise
    except Exception as e:
        raise

def test_gloo_all_to_all_support():
    """
    Test case to verify the support status of all_to_all with GLOO backend.
    Leverages the availability check pattern similar to torch.backends.nnpack.is_available.
    """
    # Check availability using the pattern from the similar API
    if not is_distributed_available():
        pytest.skip("Distributed backend is not available")

    world_size = 2
    
    # Use multiprocessing to simulate the distributed environment
    ctx = mp.get_context('spawn')
    with ctx.Pool(world_size) as pool:
        results = pool.map(run_gloo_all_to_all, range(world_size))

    # Assert that the operation is NOT supported, confirming the bug report's findings
    # that the documentation is currently incorrect.
    for res in results:
        assert res == "NOT_SUPPORTED", (
            "Expected RuntimeError: Backend gloo does not support alltoall. "
            "If this test fails, the feature may have been implemented and docs should be updated."
        )

if __name__ == "__main__":
    test_gloo_all_to_all_support()