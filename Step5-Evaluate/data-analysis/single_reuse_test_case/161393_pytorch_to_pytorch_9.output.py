import torch
import torch.distributed as dist
import tempfile
import os

def setup_distributed():
    """Initialize a single-process distributed environment for testing."""
    if not dist.is_initialized():
        # Create a temporary file for the init_method
        tmpfile = tempfile.NamedTemporaryFile(delete=False)
        tmpfile.close()
        
        try:
            dist.init_process_group(
                backend='gloo',
                init_method=f'file://{tmpfile.name}',
                world_size=1,
                rank=0
            )
            return tmpfile.name
        except Exception as e:
            print(f"Failed to initialize process group: {e}")
            os.remove(tmpfile.name)
            return None
    return None

def cleanup_distributed(tmpfile_name):
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
    if tmpfile_name and os.path.exists(tmpfile_name):
        os.remove(tmpfile_name)

# Replicate the configuration from the bug report
# Check if torch._dynamo is available to prevent AttributeError
HAS_DYNAMO = hasattr(torch, '_dynamo')
if HAS_DYNAMO:
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

def f(x):
    # Adaptation: Use torch.distributed.broadcast_object_list instead of just slicing
    # We wrap the tensor in a list to broadcast it
    object_list = [x]
    
    # Call the similar API
    dist.broadcast_object_list(object_list, src=0)
    
    # Retrieve the tensor and perform the slicing operation that caused the bug in the original issue
    # This checks if slicing works correctly after the broadcast operation within a compiled graph
    res = object_list[0]
    return res[:-1]

def main():
    if not HAS_DYNAMO:
        print("Skipping test: torch._dynamo is not available in this environment.")
        return

    tmpfile_name = setup_distributed()
    if tmpfile_name is None:
        print("Skipping test: Distributed setup failed.")
        return

    try:
        # Compile the function with fullgraph=True as in the original bug report
        compiled_f = torch.compile(f, fullgraph=True)
        
        input_tensor = torch.randn(3, 4)
        
        # Execute the compiled function
        out = compiled_f(input_tensor)
        
        print("Output shape:", out.shape)
        print("Output:", out)
        
        # Assertions to verify correctness
        assert out.shape == (2, 4), f"Expected shape (2, 4), got {out.shape}"
        assert torch.equal(out, input_tensor[:-1]), "Output values do not match expected slice"
        
        print("Test passed successfully.")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    finally:
        cleanup_distributed(tmpfile_name)

if __name__ == "__main__":
    main()