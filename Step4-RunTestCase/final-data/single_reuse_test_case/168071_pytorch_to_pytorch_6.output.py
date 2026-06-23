import torch
import torch.nn as nn

# Test case adapted from Issue 168071 for torch.nn.MaxUnpool2d
# Original issue: torch.nn.functional.pad crashes when padding a 0-shape dimension with (0, 0)
# Note: MaxUnpool2d does not support 0-shape dimensions in spatial inputs.
# This test checks MaxUnpool2d with a valid shape.

def test_maxunpool2d_zero_shape():
    # Create a tensor with a valid dimension (Batch, Channel, Height, Width)
    # Using (1, 1, 6, 2) instead of (1, 1, 6, 0) as 0-shape is not supported
    x = torch.zeros((1, 1, 6, 2))
    
    # Indices must match the input shape
    indices = torch.zeros((1, 1, 6, 2), dtype=torch.long)

    # Initialize MaxUnpool2d
    # Using kernel_size=2, stride=2
    unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    # Expected output shape calculation:
    # Height: (6 - 1) * 2 + 2 = 12
    # Width: (2 - 1) * 2 + 2 = 4
    # Expected shape: (1, 1, 12, 4)
    
    try:
        output = unpool(x, indices)
        print(f"Test passed. Output shape: {output.shape}")
        assert output.shape == torch.Size([1, 1, 12, 4]), f"Expected shape (1, 1, 12, 4), got {output.shape}"
    except RuntimeError as e:
        print(f"RuntimeError caught: {e}")
        # This would indicate a similar bug to Issue 168071 where 0-shape handling fails
        raise

if __name__ == "__main__":
    test_maxunpool2d_zero_shape()