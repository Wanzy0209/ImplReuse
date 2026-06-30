import torch
import torch.nn.functional as F
import pytest

def test_pad_circular_4d_5d_support():
    """
    Test case to verify that torch.nn.functional.pad with mode='circular'
    works for 4D and 5D inputs, as implied by the error message in issue #160053.
    
    The error message states: "Only 2D, 3D, 4D, 5D padding with non-constant 
    padding are supported for now", but the implementation previously raised
    NotImplementedError for 4D and 5D inputs.
    """
    
    # Test 4D input
    # Input shape: (2, 2, 2, 2)
    input_4d = torch.empty(2, 2, 2, 2)
    # Padding: (1, 1, 1, 1) on the last two dimensions (H, W)
    # Expected shape: (2, 2, 4, 4)
    try:
        output_4d = F.pad(input_4d, (1, 1, 1, 1), mode="circular")
        assert output_4d.shape == (2, 2, 4, 4), \
            f"Expected shape (2, 2, 4, 4) for 4D input, got {output_4d.shape}"
    except NotImplementedError as e:
        pytest.fail(f"4D circular padding failed with NotImplementedError: {e}")

    # Test 5D input
    # Input shape: (2, 2, 2, 2, 2)
    input_5d = torch.empty(2, 2, 2, 2, 2)
    # Padding: (1, 1, 1, 1, 1, 1) on the last three dimensions (D, H, W)
    # Expected shape: (2, 2, 4, 4, 4)
    try:
        output_5d = F.pad(input_5d, (1, 1, 1, 1, 1, 1), mode="circular")
        assert output_5d.shape == (2, 2, 4, 4, 4), \
            f"Expected shape (2, 2, 4, 4, 4) for 5D input, got {output_5d.shape}"
    except NotImplementedError as e:
        pytest.fail(f"5D circular padding failed with NotImplementedError: {e}")

if __name__ == "__main__":
    test_pad_circular_4d_5d_support()
    print("Test passed.")