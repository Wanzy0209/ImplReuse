import torch

# Define the function to be compiled, using torch.any
def any_op(x, y):
    # Perform a comparison and check if any element is True
    return torch.any(x > y)

def run_with_size(compiled_any, N, device, dtype):
    """Run torch.any with a specific size, creating a captured buffer sized by N."""
    # Create captured buffer that depends on dynamic N
    buffer = torch.randn(N, device=device, dtype=dtype)

    print(f"  Running with N={N}, buffer.shape={buffer.shape}")

    # Run multiple iterations with the same buffer
    for i in range(5):
        x = torch.randn(N, device=device, dtype=dtype)
        
        # Call the compiled function
        # Original call: compiled_fa(q, k, v, score_mod=..., block_mask=...)
        # Adapted call: compiled_any(x, buffer)
        output = compiled_any(x, buffer)

        # Basic assertion to ensure output is valid
        assert isinstance(output, torch.Tensor) or isinstance(output, bool)
        if isinstance(output, torch.Tensor):
            assert output.ndim == 0

    print(f"   Completed {i+1} iterations")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    torch.manual_seed(0)

    # Test with different sizes - this makes N a dynamic dimension
    # and the captured buffer (buffer) changes size with N
    sizes = [4, 8, 4, 16, 4]

    # Check if torch.compile is available (introduced in PyTorch 2.0)
    if hasattr(torch, 'compile'):
        print("torch.compile is available. Compiling function with fullgraph=True, dynamic=True...")
        compiled_any = torch.compile(any_op, fullgraph=True, dynamic=True)
    else:
        print("torch.compile is not available (requires PyTorch 2.0+). Running without compilation.")
        compiled_any = any_op

    print(f"Running torch.any with dynamic sizes on {device}, dtype={dtype}")
    print(f"Testing sizes: {sizes}\n")

    for iteration, N in enumerate(sizes, start=1):
        print(f"Iteration {iteration}:")
        run_with_size(compiled_any, N, device, dtype)

if __name__ == "__main__":
    main()