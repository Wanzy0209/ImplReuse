import torch
import torch.distributed as dist
import os
import tempfile

def setup_distributed():
    """
    Initializes the distributed process group using a temporary file for the init method
    to ensure the test is self-contained and runnable.
    """
    if not dist.is_initialized():
        # Use a temporary file for the shared store to avoid port conflicts
        with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
            url = f"file://{tmp_file.name}"
            # Initialize with a single rank (world_size=1) for a minimal reprocer
            dist.init_process_group(
                backend="gloo", 
                init_method=url, 
                rank=0, 
                world_size=1
            )

def test_gather_object_with_compile():
    """
    Test case to verify torch.distributed.gather_object behavior 
    when used with torch.compile and specific tensor dimensions 
    (d=65) from the original bug report.
    """
    setup_distributed()
    
    # Use the specific dimension 'd' that triggered the bug in the original issue
    d = 65
    
    # Create an object to gather. 
    # The original bug involved a Conv2d output with specific dimensions.
    # We mimic the data structure here.
    obj_to_gather = torch.randn((1, d, 31, 31))
    
    # Define the function containing the gather_object call
    def gather_fn(obj):
        output = [None] * dist.get_world_size()
        # gather_object collects picklable objects from the whole group
        dist.gather_object(obj, output if dist.get_rank() == 0 else None, dst=0)
        return output

    # Check if torch.compile is available (introduced in PyTorch 2.0)
    if hasattr(torch, 'compile'):
        compiled_gather = torch.compile(gather_fn)
    else:
        # Fallback for older PyTorch versions: run the function directly
        print("Warning: torch.compile is not available (requires PyTorch >= 2.0). Running without compilation.")
        compiled_gather = gather_fn

    # Attempt to run the function
    try:
        result = compiled_gather(obj_to_gather)
        
        # Verify the result
        if dist.get_rank() == 0:
            assert result is not None
            assert len(result) == 1
            assert torch.equal(result[0], obj_to_gather)
            print("Test passed: torch.distributed.gather_object works with torch.compile and d=65")
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    test_gather_object_with_compile()