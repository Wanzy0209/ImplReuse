import torch
import torch._refs
import sys

# This test assumes a 'privateuse1' backend is registered and
# implements the minimal set of ops with cpu_fallback as described
# in the bug report.

def test_atan_cpu_fallback():
    # Check if the privateuse1 backend is registered before running the test
    try:
        # Attempt to create a dummy tensor to verify backend availability
        _ = torch.empty(0, device='privateuse1')
    except RuntimeError as e:
        if "Invalid device string" in str(e):
            print("SKIP: 'privateuse1' backend is not registered. Skipping test.")
            return
        raise

    # Create a tensor on the custom device
    t = torch.randn(4, 4, device='privateuse1')
    
    # Test torch.atan (similar to torch.abs)
    # Bug: Operations like abs return empty tensors (shape [0]) when run through cpu_fallback
    result = torch.atan(t)
    
    # Assert that the shape is preserved and not empty [0]
    assert result.shape == t.shape, (
        f"torch.atan returned tensor with shape {result.shape} "
        f"instead of {t.shape} when run through cpu_fallback"
    )

    # Verify correctness of values against CPU implementation
    expected = torch.atan(t.cpu())
    assert torch.allclose(result.cpu(), expected)

if __name__ == "__main__":
    test_atan_cpu_fallback()