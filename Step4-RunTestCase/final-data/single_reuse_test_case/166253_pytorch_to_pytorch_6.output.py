import torch

# torch.compile is only available in PyTorch 2.0+
if hasattr(torch, 'compile'):
    def func_nojit(x):
        return torch.argmin(x)

    func_jit = torch.compile(func_nojit)

    # Test data with different argmin results
    # x1 has minimum at index 0
    x1 = torch.tensor([5.0, 10.0], dtype=torch.float64)
    # x2 has minimum at index 1
    x2 = torch.tensor([10.0, 5.0], dtype=torch.float64)

    # Verify non-compiled behavior
    assert func_nojit(x1) == 0
    assert func_nojit(x2) == 1

    # Verify compiled behavior
    # This checks if torch.compile correctly handles the changing input tensor
    # and does not cache the result from the first call (similar to the torch.full bug).
    assert func_jit(x1) == 0
    assert func_jit(x2) == 1

    print("Test passed.")
else:
    print("Test skipped: torch.compile is not available (requires PyTorch 2.0+).")