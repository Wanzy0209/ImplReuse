import torch
import torch.distributed as dist
import os
import tempfile

def setup():
    # Initialize the process group
    if not dist.is_initialized():
        # Use a temporary file for initialization to allow single-node execution
        with tempfile.NamedTemporaryFile(delete=True) as f:
            init_method = f"file://{f.name}"
            # For a minimal runnable test, we assume a single process (rank 0, world size 1)
            # In a real distributed scenario, this would be launched via torchrun/mp.spawn
            try:
                dist.init_process_group(
                    backend="gloo", 
                    init_method=init_method, 
                    rank=0, 
                    world_size=1
                )
            except Exception as e:
                print(f"Skipping distributed test as backend might not be available: {e}")

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def test_distributed_reduce_compile():
    setup()
    try:
        if not dist.is_initialized():
            print("Distributed not initialized, skipping test.")
            return

        # Check if torch.compile is available (requires PyTorch 2.0+)
        if not hasattr(torch, 'compile'):
            print("torch.compile is not available (requires PyTorch 2.0+), skipping test.")
            return

        # Replicate tensor shapes from the bug report
        # arg0: size=(27, 26, 62, 122)
        arg0 = torch.rand([27, 26, 62, 122], dtype=torch.float32, device='cpu', requires_grad=True)
        # arg1: size=(27, 26, 124, 122)
        arg1 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device='cpu', requires_grad=True)
        # arg2: size=(27, 26, 124, 122)
        arg2 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device='cpu', requires_grad=True)
        
        # Define the function to be tested
        # Original: t3 = flex_attention(t0, t1, t2)
        # Adapted: Use torch.distributed.reduce on the first tensor
        def foo(t0, t1, t2):
            # flex_attention returns a new tensor, reduce is in-place.
            # We clone t0 to mimic the output creation and reduce it.
            t3 = t0.clone()
            # Perform the reduction operation (sum is default)
            torch.distributed.reduce(t3, dst=0)
            return t3

        # 1. Run in Eager mode
        # Note: In a single-process world_size=1, reduce is a no-op (tensor remains same)
        eager_output = foo(arg0, arg1, arg2)
        
        # 2. Run in Compiled mode
        # This checks if torch.compile handles the distributed call correctly
        compiled_foo = torch.compile(foo)
        compiled_output = compiled_foo(arg0, arg1, arg2)

        # 3. Verify results
        # Since world_size is 1, reduce shouldn't change the tensor values
        assert torch.allclose(eager_output, compiled_output), "Eager and Compiled outputs differ"
        
        print("Test passed: torch.distributed.reduce works with torch.compile")

    finally:
        cleanup()

if __name__ == "__main__":
    test_distributed_reduce_compile()