import torch
import torch.distributed as dist
import tempfile
import os
import sys

def test_gather_object_compile_with_debug_mode():
    """
    Test case to verify interaction between torch.compile, DebugMode, 
    and torch.distributed.gather_object.
    
    Based on Issue 164143: DebugMode silently disables torch.compile.
    This test adapts the scenario to use torch.distributed.gather_object.
    """
    
    # Setup minimal distributed environment (single process for simplicity)
    # Using a temporary file for the init method
    with tempfile.NamedTemporaryFile(delete=False) as f:
        init_method = f"file://{f.name}"

    try:
        # Initialize process group
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=0,
            world_size=1
        )

        # Define a function that includes the similar API: torch.distributed.gather_object
        def func_to_compile(x):
            # Perform a tensor operation to give torch.compile something to optimize
            y = x + 1
            
            # Call the similar API
            # Since world_size is 1 and rank is 0 (dst), we provide a list
            output_list = [None] * 1
            dist.gather_object(y.item(), output_list, dst=0)
            
            return y

        # Import DebugMode as referenced in the bug report
        # Note: This is an internal testing utility.
        try:
            from torch.testing._internal.debug_mode import DebugMode
        except ImportError:
            print("Skipping test: torch.testing._internal.debug_mode not available in this build.")
            return

        # The bug report suggests using 'aot_eager' as a backend that might work
        # or at least should be tested against.
        backend = "aot_eager"

        print(f"Testing torch.compile with backend={backend} inside DebugMode...")
        
        with DebugMode():
            # Attempt to compile the function
            compiled_func = torch.compile(func_to_compile, backend=backend)
            
            input_tensor = torch.tensor(1.0)
            
            # Execute
            # If the bug is present (silent disable), this runs but isn't compiled.
            # If the fix (error) is present, this might raise an error.
            # If the fix (support) is present, this runs compiled.
            result = compiled_func(input_tensor)
            
            # Basic assertion to ensure execution completed
            assert torch.allclose(result, input_tensor + 1)
            print("Execution completed.")

    finally:
        # Cleanup
        dist.destroy_process_group()
        os.unlink(init_method.replace("file://", ""))

if __name__ == "__main__":
    test_gather_object_compile_with_debug_mode()