import torch
import torch.distributed as dist
import os
import tempfile

def test_torch_distributed_new_group():
    """
    Test case for torch.distributed.new_group.
    
    Note: The original bug report (Issue 163337) describes a compilation error 
    specific to torch.utils.cpp_extension.load on ROCm. Since 
    torch.distributed.new_group is a runtime API for process group management 
    and does not involve compiling C++ extensions, the specific compilation 
    error cannot be reproduced here. This test verifies the basic functionality 
    of the similar API.
    """
    
    # Initialize the process group if not already initialized.
    # We use 'gloo' backend for CPU compatibility to ensure the test is runnable
    # without requiring specific hardware like ROCm or CUDA.
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        
        # Using a temporary file for the store to ensure clean initialization
        with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
            dist.init_process_group(
                backend='gloo',
                init_method=f'file://{tmp_file.name}',
                rank=0,
                world_size=1
            )

    # Adapted call site: replacing torch.utils.cpp_extension.load with torch.distributed.new_group
    # Create a new group containing only the current rank (rank 0)
    ranks = [0]
    new_group = dist.new_group(ranks=ranks)

    # Assertions to verify the group was created correctly
    assert new_group is not None, "New group creation returned None"
    assert new_group.size() == 1, f"Expected group size 1, got {new_group.size()}"
    assert new_group.rank() == 0, f"Expected group rank 0, got {new_group.rank()}"

    print("Test passed: torch.distributed.new_group executed successfully.")

if __name__ == "__main__":
    test_torch_distributed_new_group()