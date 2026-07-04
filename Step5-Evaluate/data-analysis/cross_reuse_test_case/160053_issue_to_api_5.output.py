import torch
import torch.nn.functional as F
import pytest

def test_pad_circular_higher_dimensions():
    """
    Test case for Issue #160053.
    
    The bug report indicates that torch.nn.functional.pad with mode="circular"
    raises a NotImplementedError for 4D and 5D inputs, despite the error message
    claiming that "Only 2D, 3D, 4D, 5D padding ... are supported".
    
    This test verifies that circular padding works correctly for the dimensions
    explicitly claimed as supported in the error message.
    """
    
    # Test cases based on the dimensions mentioned in the error message
    # Format: (dimensions, tensor_shape)
    test_cases = [
        (2, (2, 2)),
        (3, (2, 2, 2)),
        (4, (2, 2, 2, 2)), # Specifically failing in the bug report
        (5, (2, 2, 2, 2, 2)) # Specifically failing in the bug report
    ]

    for ndim, shape in test_cases:
        # Create a random tensor of the target dimension
        tensor = torch.randn(shape)
        
        # Padding configuration
        # Fix: circular padding requires padding to be specified for all dimensions.
        # The padding tuple format is (pad_left, pad_right, pad_top, pad_bottom, ...).
        # We pad 1 on both sides for each dimension.
        padding = tuple([1, 1] * ndim)
        
        # Calculate expected output shape
        # Since we pad 1 on both sides for all dimensions, each dimension increases by 2.
        expected_shape = [s + 2 for s in shape]

        # Attempt to pad with circular mode
        # We expect this to succeed based on the API's error message claims.
        try:
            result = F.pad(tensor, padding, mode="circular")
            assert result.shape == torch.Size(expected_shape), \
                f"Shape mismatch for {ndim}D tensor. Expected {expected_shape}, got {result.shape}"
        except NotImplementedError as e:
            error_msg = str(e)
            # Check if the error message contradicts itself (the original bug)
            if f"{ndim}D" in error_msg and "supported" in error_msg:
                pytest.fail(
                    f"Bug reproduced for {ndim}D input: Operation failed with "
                    f"NotImplementedError, but error message claims support: {e}"
                )
            else:
                # If it fails for a reason other than the misleading message, re-raise
                raise

if __name__ == "__main__":
    test_pad_circular_higher_dimensions()
    print("Test passed.")