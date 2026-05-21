import torch
import torch.nn.functional as F

# Test case for torch.nn.functional.conv3d handling 0-shape dimensions
# Adapted from the torch.nn.functional.pad issue where padding (0, 0) on a 0-shape dimension caused a crash.

def test_conv3d_zero_shape():
    # Create a 5D input tensor with a 0 dimension (Depth is 0)
    # Shape: (Batch, Channel, Depth, Height, Width)
    x = torch.zeros((1, 3, 0, 5, 5))

    # Create weights
    # Shape: (Out_Channel, In_Channel, kD, kH, kW)
    # kD is set to 1 to ensure the output size is 0 (not negative) given input size 0 and padding 0.
    # Formula: output = floor((input + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
    # With input=0, padding=0, dilation=1, kernel=1, stride=1 -> output = 0.
    weight = torch.randn(3, 3, 1, 3, 3)

    # Apply conv3d
    # Padding format: (pad_left_d, pad_right_d, pad_left_h, pad_right_h, pad_left_w, pad_right_w)
    # We specifically pad the 0-dimension (Depth) with (0, 0) to mimic the bug scenario.
    try:
        output = F.conv3d(x, weight, padding=(0, 0, 0, 0, 0, 0))
        print("Output shape:", output.shape)
        
        # Expected shape: (1, 3, 0, 3, 3)
        # The depth dimension should remain 0.
        assert output.shape == torch.Size([1, 3, 0, 3, 3]), f"Expected shape [1, 3, 0, 3, 3], got {output.shape}"
        print("Test passed.")
    except RuntimeError as e:
        print(f"RuntimeError encountered: {e}")
        print("Test failed (similar to original pad bug).")

if __name__ == "__main__":
    test_conv3d_zero_shape()