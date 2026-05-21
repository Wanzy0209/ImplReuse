import torch
import pytest

def test_torch_prod_compile_dim0():
    """
    Test case for torch.prod based on the torch.compile regression report.
    The original issue involves a performance regression in torch.compile
    for operations along rows (dim 0) with specific tensor dimensions.
    This test verifies the correctness of torch.compile applied to torch.prod
    along dimension 0 with similar tensor shapes.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test")

    # Dimensions from the bug report (M=16384, K=16384)
    M, K = 16384, 16384
    
    # Create input tensor on GPU
    x = torch.randn(M, K, device='cuda', dtype=torch.float32)

    # Define the function using the similar API: torch.prod
    # The bug report highlights operations along rows (dim 0)
    def prod_func(x):
        return torch.prod(x, dim=0)

    # Compile the function using torch.compile (the API under test in the bug)
    compiled_prod_func = torch.compile(prod_func)

    # Execute both eager and compiled versions
    expected = prod_func(x)
    actual = compiled_prod_func(x)

    # Assert correctness to ensure the compilation didn't break logic
    # (Performance regressions are harder to test in unit tests, so we check correctness)
    assert torch.allclose(expected, actual, rtol=1e-3, atol=1e-3), \
        "Compiled torch.prod output differs from eager output"

if __name__ == "__main__":
    test_torch_prod_compile_dim0()
    print("Test passed.")