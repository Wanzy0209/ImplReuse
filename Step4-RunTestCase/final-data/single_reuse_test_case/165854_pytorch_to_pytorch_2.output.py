import torch
import sys

# Fix for AttributeError: module 'torch' has no attribute 'compile'
# This happens in PyTorch versions < 2.0. We mock it to allow the test to run in eager mode.
if not hasattr(torch, 'compile'):
    print("Warning: torch.compile not found (PyTorch < 2.0). Using eager mode (pass-through mock).")
    def compile_mock(func, **kwargs):
        return func
    torch.compile = compile_mock


def run_with_size(compiled_fn, N, device, dtype):
    """Run prod with a specific size, creating a buffer sized by N."""
    # Create captured buffer that depends on dynamic N
    scale = torch.randn(N, device=device, dtype=dtype, requires_grad=True)

    print(f"  Running with N={N}, scale.shape={scale.shape}")

    # Run multiple iterations with the same scale
    for i in range(5):
        x = torch.randn(N, device=device, dtype=dtype, requires_grad=True)

        # Call the compiled function
        # Original call: compiled_fa(q, k, v, score_mod=score_mod, block_mask=block_mask)
        # Adapted call: compiled_fn(x, scale)
        outputs = compiled_fn(x, scale)
        loss = outputs.sum()
        loss.backward()

    print(f"   Completed {i+1} iterations")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    torch.manual_seed(0)

    # Test with different sizes - this makes N a dynamic dimension
    # and the captured buffer (scale) changes size with N
    sizes = [4, 8, 4, 16, 4]

    # Define the function to be compiled
    # Mimics the interaction between input and a dynamic buffer using torch.prod
    def prod_op(x, scale):
        return torch.prod(x * scale)

    compiled_prod = torch.compile(prod_op, fullgraph=True, dynamic=True)

    print(f"Running torch.prod with dynamic sizes on {device}, dtype={dtype}")
    print(f"Testing sizes: {sizes}\n")

    for iteration, N in enumerate(sizes, start=1):
        print(f"Iteration {iteration}:")
        run_with_size(compiled_prod, N, device, dtype)


if __name__ == "__main__":
    main()