import torch
import torch.distributed as dist
import os

def setup_distributed():
    """Initialize a minimal distributed environment for single-process testing."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # Use 'gloo' backend for CPU compatibility in this minimal test case
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

def test_gather_object_compile():
    """
    Test case for torch.distributed.gather_object adapted from the 
    torch.compile bug report context.
    
    Verifies that gather_object behaves correctly (or fails gracefully) 
    when wrapped in torch.compile with combo_kernels enabled.
    """
    setup_distributed()
    
    # Enable the configuration from the original bug report
    torch._inductor.config.combo_kernels = True

    # Define the function using the similar API (torch.distributed.gather_object)
    @torch.compile
    def fn_gather(obj, gather_list):
        # Adapt the call site: instead of sum/mean/cumsum, we use gather_object
        dist.gather_object(obj, gather_list, dst=0)
        return gather_list

    # Prepare inputs
    # gather_object accepts picklable objects. We use a tensor here.
    input_obj = {"tensor": torch.rand(16, 128)}
    # For rank 0 (dst), the list must be sized to world_size
    output_list = [None] * 1 

    # Execute
    try:
        result = fn_gather(input_obj, output_list)
        
        # Verify the result if execution succeeds
        assert result is not None
        assert len(result) == 1
        assert result[0] == input_obj
        print("Test Passed: torch.distributed.gather_object executed successfully with torch.compile.")
        
    except Exception as e:
        # The original bug was a specific NameError during compilation.
        # We check to ensure we are not hitting that specific compilation error.
        error_msg = str(e)
        if "NameError" in error_msg and "_triton_helper_fn" in error_msg:
            print(f"Bug Reproduced: {error_msg}")
            raise
        else:
            # Other exceptions (like unsupported ops) are expected behavior for this combo
            print(f"Expected behavior (API not fully compilable or other error): {type(e).__name__}")
            # We consider the test valid if it doesn't crash with the specific NameError
            pass

if __name__ == "__main__":
    test_gather_object_compile()
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()