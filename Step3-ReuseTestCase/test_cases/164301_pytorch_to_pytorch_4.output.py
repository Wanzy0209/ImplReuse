import torch
import sys

def test_torch_all_compile():
    """
    Test case to verify torch.all works correctly with torch.compile.
    Adapted from the context of Issue 164301 regarding torch.compile regressions.
    """
    # Check for CUDA availability as the original bug was specific to GPU (B200)
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    print(f"Running test on GPU: {torch.cuda.get_device_name(0)}")
    
    # Define a function using torch.all
    # The original bug involved operations along rows (dim0), so we test reduction.
    def all_func(x):
        return torch.all(x)

    # Compile the function
    compiled_all_func = torch.compile(all_func)

    # Create input tensor
    # Using dimensions similar to the bug report (M=16384, K=16384) to stress the compiler,
    # though using standard float32 for compatibility without torchao.
    x = torch.randn(16384, 16384, dtype=torch.float32, device='cuda')

    # Execute eager mode
    expected = all_func(x)

    # Execute compiled mode
    actual = compiled_all_func(x)

    # Verify correctness
    assert expected.dtype == actual.dtype, f"Dtype mismatch: expected {expected.dtype}, got {actual.dtype}"
    assert expected.item() == actual.item(), f"Value mismatch: expected {expected.item()}, got {actual.item()}"

    # Test with dimension argument (reduction along dim 0)
    def all_func_dim(x):
        return torch.all(x, dim=0)

    compiled_all_func_dim = torch.compile(all_func_dim)
    expected_dim = all_func_dim(x)
    actual_dim = compiled_all_func_dim(x)

    assert torch.equal(expected_dim, actual_dim), "Mismatch in reduction along dim 0"

    print("Test passed: torch.compile works correctly with torch.all")

if __name__ == "__main__":
    test_torch_all_compile()