import torch
import torch.nn.functional as F
import sys

def test_pad_circular_high_dimensions():
    """
    Test that torch.nn.functional.pad with mode='circular' works for 4D and 5D inputs.
    This addresses the issue where an NotImplementedError was raised despite the 
    error message claiming support for these dimensions.
    """
    # Check PyTorch version to skip if the fix is not yet present (mimicking the pattern of the similar API)
    # Note: The bug was reported in 2.7.1. We assume the test runs on a version where this might be fixed or needs verification.
    # If the version is strictly older than a hypothetical fix version, we might expect the error, 
    # but here we assert the correct behavior.
    
    # Test 4D Input
    # Shape: (Batch, Channel, Height, Width) -> (2, 2, 2, 2)
    input_4d = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2)
    
    # Pad the last dimension (Width) by 1 on left and 1 on right
    # Expected Shape: (2, 2, 2, 4)
    try:
        output_4d = F.pad(input_4d, (1, 1), mode="circular")
        assert output_4d.shape == (2, 2, 2, 4), f"Expected shape (2, 2, 2, 4), got {output_4d.shape}"
        
        # Verify circular wrapping logic
        # The last element of the original dimension should appear at the start of the padding
        assert torch.equal(output_4d[..., 0], input_4d[..., -1]), "Circular padding failed for 4D input (left side)"
        assert torch.equal(output_4d[..., -1], input_4d[..., 0]), "Circular padding failed for 4D input (right side)"
        
    except NotImplementedError as e:
        print(f"Skipping 4D test as functionality is not implemented: {e}")
        # Depending on the test framework, we might re-raise or mark as xfail
        raise

    # Test 5D Input
    # Shape: (Batch, Time, Channel, Height, Width) -> (2, 2, 2, 2, 2)
    input_5d = torch.arange(32, dtype=torch.float32).reshape(2, 2, 2, 2, 2)
    
    # Pad the last dimension (Width) by 1 on left and 1 on right
    # Expected Shape: (2, 2, 2, 2, 4)
    try:
        output_5d = F.pad(input_5d, (1, 1), mode="circular")
        assert output_5d.shape == (2, 2, 2, 2, 4), f"Expected shape (2, 2, 2, 2, 4), got {output_5d.shape}"
        
        # Verify circular wrapping logic
        assert torch.equal(output_5d[..., 0], input_5d[..., -1]), "Circular padding failed for 5D input (left side)"
        assert torch.equal(output_5d[..., -1], input_5d[..., 0]), "Circular padding failed for 5D input (right side)"
        
    except NotImplementedError as e:
        print(f"Skipping 5D test as functionality is not implemented: {e}")
        raise

if __name__ == "__main__":
    test_pad_circular_high_dimensions()
    print("Test passed successfully.")