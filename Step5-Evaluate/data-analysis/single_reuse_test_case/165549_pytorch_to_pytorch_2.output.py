import torch

def test_neg_cpu_fallback():
    # Check if the custom backend is available before running the test
    try:
        # Attempt to create a dummy tensor to verify device availability
        _ = torch.empty(1, device='privateuse1')
    except RuntimeError as e:
        if "Invalid device string" in str(e):
            print("Skipping test: 'privateuse1' backend is not registered.")
            return
        raise

    # Create a tensor on the custom device
    t = torch.randn(4, 4, device='privateuse1')
    
    # Call the similar API (torch.neg)
    # The bug report indicates that torch.neg works correctly (uses structured_delegate)
    # unlike torch.abs which returns an empty tensor.
    result = torch.neg(t)
    
    # Verify the result shape is correct (not empty)
    assert result.shape == torch.Size([4, 4]), f"Expected shape [4, 4], but got {result.shape}"
    
    # Verify the values are correct
    expected = -t.cpu()
    assert torch.allclose(result.cpu(), expected)

if __name__ == "__main__":
    test_neg_cpu_fallback()