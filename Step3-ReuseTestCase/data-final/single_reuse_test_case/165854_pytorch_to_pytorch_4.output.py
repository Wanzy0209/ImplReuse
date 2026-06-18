import torch


def run_with_size(compiled_all, N, device, dtype):
    """Run torch.all with a specific size, creating a tensor sized by N."""
    # Create a tensor that depends on dynamic N
    # Mimicking the dynamic input aspect of the original bug
    x = torch.randn(N, device=device, dtype=dtype, requires_grad=True)

    print(f"  Running with N={N}, x.shape={x.shape}")

    # Run multiple iterations with the same size
    for i in range(5):
        # Re-generate x to ensure gradients are fresh for the iteration
        x = torch.randn(N, device=device, dtype=dtype, requires_grad=True)
        
        # Call the compiled function with a condition
        # torch.all returns a boolean scalar tensor
        outputs = compiled_all(x > 0)
        
        # Perform backward pass
        # Cast to float because backward is not supported on boolean tensors
        loss = outputs.float()
        loss.backward()

    print(f"   Completed {i+1} iterations")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    torch.manual_seed(0)

    # Test with different sizes - this makes N a dynamic dimension
    sizes = [4, 8, 4, 16, 4]

    # Define the function to be compiled
    def all_func(x):
        return torch.all(x)

    # Compile with dynamic=True to handle varying input sizes
    compiled_all = torch.compile(all_func, fullgraph=True, dynamic=True)

    print(f"Running torch.all with dynamic sizes on {device}, dtype={dtype}")
    print(f"Testing sizes: {sizes}\n")

    for iteration, N in enumerate(sizes, start=1):
        print(f"Iteration {iteration}:")
        run_with_size(compiled_all, N, device, dtype)


if __name__ == "__main__":
    main()