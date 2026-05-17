import torch
import torch.nn.functional as F

def test_conv1d_negative_padding():
    """
    Test case adapted from the constant_pad_nd negative padding bug report.
    Verifies that torch.nn.functional.conv1d handles negative padding correctly,
    specifically checking for valid cropping, zero-size outputs, and invalid negative sizes.
    """
    
    # Setup input and weight tensors
    # Input shape: (Batch=1, Channels=1, Width=5)
    input_tensor = torch.ones([1, 1, 5])
    # Weight shape: (OutChannels=1, InChannels=1, KernelSize=3)
    weight = torch.ones([1, 1, 3])

    # Case 1: Valid negative padding (cropping)
    # Padding -1 crops 1 from both sides. Effective width = 5 - 2 = 3.
    # Output width = (3 - 3) / 1 + 1 = 1.
    output = F.conv1d(input_tensor, weight, padding=-1)
    assert output.shape == torch.Size([1, 1, 1]), f"Expected shape [1, 1, 1], got {output.shape}"

    # Case 2: Negative padding resulting in size 0
    # Mirrors the behavior in the bug report where output size becomes 0.
    # Input width 5, Kernel 4, Padding -1.
    # Output width = (5 + 2*(-1) - 4 + 1) = 0.
    weight_4 = torch.ones([1, 1, 4])
    output_zero = F.conv1d(input_tensor, weight_4, padding=-1)
    assert output_zero.shape == torch.Size([1, 1, 0]), f"Expected shape [1, 1, 0], got {output_zero.shape}"

    # Case 3: Negative padding resulting in negative size (should raise error)
    # Mirrors the error condition in the bug report.
    # Input width 5, Kernel 4, Padding -2.
    # Output width = (5 + 2*(-2) - 4 + 1) = -2.
    try:
        F.conv1d(input_tensor, weight_4, padding=-2)
        raise AssertionError("Expected RuntimeError for negative output size")
    except RuntimeError as e:
        # Check if the error message is related to size calculation
        assert "negative output size" in str(e) or "too small" in str(e), f"Unexpected error message: {e}"

if __name__ == "__main__":
    test_conv1d_negative_padding()
    print("Test passed.")