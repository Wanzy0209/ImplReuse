import torch
import torch.distributed as dist
import tempfile
import os
import sys

def test_distributed_reduce_in_compile():
    """
    Test case to verify the behavior of torch.distributed.reduce inside torch.compile.
    This adapts the original issue regarding tolist() graphing to the similar API torch.distributed.reduce.
    """
    # Check if torch.compile is available (requires PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        print("torch.compile is not available (requires PyTorch 2.0+). Skipping test.")
        return

    if not dist.is_available():
        print("torch.distributed is not available. Skipping test.")
        return

    # Setup a temporary file store for single-process distributed execution
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()

    try:
        # Initialize process group for a single process (world_size=1)
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{temp_file.name}",
            rank=0,
            world_size=1
        )

        # Adapted test case: replacing a.tolist() with torch.distributed.reduce
        @torch.compile(fullgraph=False, backend="eager")
        def func(a):
            # Original call site was: u0, u1 = a.tolist()
            # Adapted call site uses torch.distributed.reduce
            # Note: reduce is an in-place operation
            dist.reduce(a, dst=0)
            return a

        # Execute the function
        input_tensor = torch.tensor([1.0, 2.0])
        output = func(input_tensor)

        # Verify execution
        # Since rank=0 is the dst, the tensor remains unchanged (sum of 1 element is itself)
        assert torch.equal(output, input_tensor), "Output tensor should match input for single-process reduce"
        print("Test passed: torch.distributed.reduce inside torch.compile executed successfully.")

    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

if __name__ == "__main__":
    test_distributed_reduce_in_compile()