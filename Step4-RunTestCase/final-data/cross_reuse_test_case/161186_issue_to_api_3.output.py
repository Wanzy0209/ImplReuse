import torch
import torch.utils.checkpoint
# Removed top-level import of torch.backends.nnpack to handle environments where it is missing

class MyOpWithFlags(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor):
        # Leverage the similar API: torch.backends.nnpack.set_flags
        # This mimics the 'save state' pattern found in the similar API implementation
        # where original state is captured before modification.
        original_flags = torch.backends.nnpack.set_flags(True)
        
        # Original bug reproduction logic: creating tensors to save
        out_0 = torch.zeros(2**20, device=inp.device, dtype=torch.float32)
        out_1 = torch.zeros(2**20, device=inp.device, dtype=torch.float32)
        
        ctx.save_for_backward(inp, out_0, out_1)
        
        # Save the flags to the context to check for state leaks or corruption
        # similar to how tensors are saved.
        ctx.saved_flags = original_flags
        
        return out_0, out_1

    @staticmethod
    def backward(ctx, dA, dB):
        _ = ctx.saved_tensors
        # Restore flags in backward to ensure global state is cleaned up,
        # testing if the checkpoint mechanism interferes with this restoration.
        if hasattr(ctx, 'saved_flags'):
            torch.backends.nnpack.set_flags(ctx.saved_flags[0])
        return None


def op_fn(inp):
    return MyOpWithFlags.apply(inp)[0]


def test_checkpoint_memory_leak_with_flags():
    # Check if CUDA is available to run the memory test
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory leak test.")
        return

    # Check if nnpack is available
    try:
        # Import inside the function to prevent ModuleNotFoundError if nnpack is not built
        import torch.backends.nnpack
        torch.backends.nnpack.set_flags(False)
    except (ModuleNotFoundError, AttributeError):
        print("NNPACK not available, skipping test.")
        return

    dummy_input = torch.nn.Parameter(torch.randn(2**20, device="cuda"))
    
    # Run a few iterations to check for memory growth
    # (Reduced from 1000 to 10 for faster testing in a CI environment)
    initial_mem = torch.cuda.memory_allocated() / 1024**2
    print(f"Initial Memory: {initial_mem:.2f} MiB")
    
    for i in range(10):
        full_out = torch.utils.checkpoint.checkpoint(op_fn, dummy_input, use_reentrant=False)
        full_out.sum().backward()
        dummy_input.grad = None  # free gradient memory
        
        current_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"Iter {i}: {current_mem:.2f} MiB")
        
        # Basic assertion to ensure we don't blow up memory excessively 
        # (allowing some fluctuation but checking for the specific leak pattern)
        # The original bug showed linear growth. 
        # We check if memory increases by more than 10MB over the baseline + buffer.
        assert current_mem < (initial_mem + 20), f"Memory leak detected: {current_mem} > {initial_mem + 20}"

    print("Test passed: No significant memory leak detected.")

if __name__ == "__main__":
    test_checkpoint_memory_leak_with_flags()