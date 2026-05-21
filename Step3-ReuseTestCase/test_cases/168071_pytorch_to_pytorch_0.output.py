import torch
import torch.nn.functional as F

def test_pad_zero_shape_with_zero_padding():
    """
    Test case for Issue #168071.
    Verifies that padding a 0-shape dimension with (0, 0) does not raise a RuntimeError
    and produces the correct output shape.
    """
    # Case 1: Original bug report scenario
    # Input shape (6, 0), pad (0, 0, 0, 24)
    # Last dim (size 0) gets (0, 0) -> 0
    # First dim (size 6) gets (0, 24) -> 30
    x0 = torch.zeros((6, 0))
    try:
        x1 = F.pad(x0, (0, 0, 0, 24))
        assert x1.shape == torch.Size([30, 0]), f"Expected shape (30, 0), got {x1.shape}"
        print("Test Case 1 Passed: Shape is correct.")
    except RuntimeError as e:
        print(f"Test Case 1 Failed with RuntimeError: {e}")

    # Case 2: 3D tensor with zero in middle dimension, padded with (0, 0)
    # Input shape (2, 0, 2), pad (0, 0, 0, 0, 0, 0)
    # All dimensions padded with 0.
    x2 = torch.zeros((2, 0, 2))
    try:
        x3 = F.pad(x2, (0, 0, 0, 0, 0, 0))
        assert x3.shape == torch.Size([2, 0, 2]), f"Expected shape (2, 0, 2), got {x3.shape}"
        print("Test Case 2 Passed: Shape is correct.")
    except RuntimeError as e:
        print(f"Test Case 2 Failed with RuntimeError: {e}")

    # Case 3: Zero shape dimension padded with non-zero values (sanity check)
    # Input shape (6, 0), pad (1, 1, 0, 0)
    # Last dim (size 0) gets (1, 1) -> 2
    # First dim (size 6) gets (0, 0) -> 6
    try:
        x4 = F.pad(x0, (1, 1, 0, 0))
        assert x4.shape == torch.Size([6, 2]), f"Expected shape (6, 2), got {x4.shape}"
        print("Test Case 3 Passed: Shape is correct.")
    except RuntimeError as e:
        print(f"Test Case 3 Failed with RuntimeError: {e}")

if __name__ == "__main__":
    test_pad_zero_shape_with_zero_padding()