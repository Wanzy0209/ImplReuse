import torch
import pytest

def test_constant_pad_nd_negative_padding_consistency():
    """
    Test for Issue 161014: Inconsistent constant_pad_nd behavior with negative padding.
    
    The op `constant_pad_nd` should allow negative padding (shrinking) as long as the 
    resulting output size is non-negative. The bug report highlights an inconsistency 
    where mixed positive and negative padding throws a RuntimeError even when the 
    resulting dimensions are valid (e.g., size 0).
    """
    input_tensor = torch.ones([5, 3])

    # Case 1: Negative padding on the last dimension only.
    # Input shape [5, 3], padding [-1, -2] -> Dim 1: 3 - 1 - 2 = 0.
    # Expected shape: [5, 0]
    out1 = torch.ops.aten.constant_pad_nd.default(input_tensor, [-1, -2])
    assert out1.shape == torch.Size([5, 0]), "Failed to shrink last dimension to 0"

    # Case 2: Negative padding on last dimension, zero padding on first.
    # Input shape [5, 3], padding [-1, -2, 0, 0] -> Dim 1: 3 - 1 - 2 = 0, Dim 0: 5 + 0 + 0 = 5.
    # Expected shape: [5, 0]
    out2 = torch.ops.aten.constant_pad_nd.default(input_tensor, [-1, -2, 0, 0])
    assert out2.shape == torch.Size([5, 0]), "Failed with zero padding on first dim and negative on last"

    # Case 3: Negative padding on last dimension, positive padding on first.
    # This is the specific failing case in the bug report.
    # Input shape [5, 3], padding [-1, -2, 1, 1] -> Dim 1: 3 - 1 - 2 = 0, Dim 0: 5 + 1 + 1 = 7.
    # Expected shape: [7, 0]
    # Bug behavior: RuntimeError: "resulted in a negative output size"
    out3 = torch.ops.aten.constant_pad_nd.default(input_tensor, [-1, -2, 1, 1])
    assert out3.shape == torch.Size([7, 0]), "Failed with mixed positive/negative padding resulting in size 0"

    # Case 4: Ensure non-zero resulting size works with mixed padding.
    # Input shape [5, 3], padding [-1, -1, 1, 1] -> Dim 1: 3 - 1 - 1 = 1, Dim 0: 5 + 1 + 1 = 7.
    # Expected shape: [7, 1]
    out4 = torch.ops.aten.constant_pad_nd.default(input_tensor, [-1, -1, 1, 1])
    assert out4.shape == torch.Size([7, 1]), "Failed with mixed padding resulting in positive size"

if __name__ == "__main__":
    test_constant_pad_nd_negative_padding_consistency()
    print("All tests passed.")