import torch
import torch.nn.functional as F

def test_prelu_non_contiguous_input():
    """
    Test case adapted from Issue 164491 regarding layout handling.
    
    The original bug report highlights that `_scaled_mm` and `_int_mm` 
    raise errors or suffer performance degradation when the right-hand side 
    matrix is row-major (or non-contiguous in a way that differs from the 
    kernel's expectation).
    
    This test verifies that `torch.nn.functional.prelu` handles non-contiguous 
    input tensors correctly without raising errors, ensuring robustness against 
    varying memory layouts.
    """
    print("Testing PReLU with non-contiguous inputs...")

    # Setup: Create a contiguous input tensor and weight
    # PReLU weight is typically 1D or shared across channels
    batch_size, channels, height, width = 4, 16, 8, 8
    x_cont = torch.randn(batch_size, channels, height, width, dtype=torch.float32)
    weight = torch.randn(channels, dtype=torch.float32)

    # 1. Baseline: Compute PReLU with contiguous input
    try:
        y_cont = F.prelu(x_cont, weight)
    except Exception as e:
        print(f"FAILED: PReLU raised error on contiguous input: {e}")
        return

    # 2. Test Case: Compute PReLU with non-contiguous input
    # We simulate a "row-major" or strided layout issue by permuting dimensions 
    # and permuting back, which results in a non-contiguous tensor with the same shape.
    x_non_cont = x_cont.permute(0, 2, 3, 1).permute(0, 3, 1, 2)
    
    assert not x_non_cont.is_contiguous(), "Test setup failed: x_non_cont should be non-contiguous"

    try:
        y_non_cont = F.prelu(x_non_cont, weight)
    except Exception as e:
        print(f"FAILED: PReLU raised error on non-contiguous input: {e}")
        return

    # 3. Verification: Ensure outputs are numerically identical
    if torch.allclose(y_cont, y_non_cont):
        print("PASSED: PReLU handles non-contiguous input correctly and produces identical results.")
    else:
        print("FAILED: PReLU output mismatch between contiguous and non-contiguous inputs.")
        print(f"Max diff: {torch.max(torch.abs(y_cont - y_non_cont))}")

if __name__ == "__main__":
    test_prelu_non_contiguous_input()