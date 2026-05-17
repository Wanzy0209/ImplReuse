import torch
import torch.nn.functional as F
import pytest

def test_constant_pad_nd_negative_padding_consistency():
    """
    Test case based on Issue 161014: Inconsistent constant_pad_nd behavior with negative padding.
    
    The bug report highlights that constant_pad_nd allows negative padding (shrinking)
    as long as the resulting size is non-negative. However, it throws an error
    when mixing negative padding (shrinking to 0) on one dimension with positive
    padding on another.
    """
    x = torch.ones([5, 3])

    # Case 1: Negative padding on the last dimension (works)
    # Input shape [5, 3], padding [-1, -2] -> Last dim: 3 - 1 - 2 = 0
    # Expected shape: [5, 0]
    out = F.pad(x, [-1, -2])
    assert out.shape == torch.Size([5, 0]), f"Expected shape [5, 0], got {out.shape}"

    # Case 2: Negative padding on last dim, zero padding on first dim (works)
    # Input shape [5, 3], padding [-1, -2, 0, 0]
    # Expected shape: [5, 0]
    out = F.pad(x, [-1, -2, 0, 0])
    assert out.shape == torch.Size([5, 0]), f"Expected shape [5, 0], got {out.shape}"

    # Case 3: Negative padding on both dimensions (works)
    # Input shape [5, 3], padding [-1, -2, -1, -1]
    # First dim: 5 - 1 - 1 = 3, Last dim: 3 - 1 - 2 = 0
    # Expected shape: [3, 0]
    out = F.pad(x, [-1, -2, -1, -1])
    assert out.shape == torch.Size([3, 0]), f"Expected shape [3, 0], got {out.shape}"

    # Case 4: Negative padding on last dim (shrinking to 0), positive padding on first dim
    # Input shape [5, 3], padding [-1, -2, 1, 1]
    # First dim: 5 + 1 + 1 = 7, Last dim: 3 - 1 - 2 = 0
    # Expected shape: [7, 0]
    # Bug report: This case throws a RuntimeError, which is inconsistent with the above cases.
    # This test asserts the correct behavior (no error).
    try:
        out = F.pad(x, [-1, -2, 1, 1])
        assert out.shape == torch.Size([7, 0]), f"Expected shape [7, 0], got {out.shape}"
    except RuntimeError as e:
        pytest.fail(f"constant_pad_nd raised an error unexpectedly for mixed padding: {e}")

if __name__ == "__main__":
    test_constant_pad_nd_negative_padding_consistency()
    print("Test passed.")