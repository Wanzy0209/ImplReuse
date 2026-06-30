import torch
import torch.nn.functional as F

def test_pad_circular_dimensions_consistency():
    """
    Test case for Issue #160053.
    
    The error message in F.pad states: "Only 2D, 3D, 4D, 5D padding with 
    non-constant padding are supported for now". However, the implementation
    raises NotImplementedError for 4D and 5D inputs.
    
    This test verifies that the implementation aligns with the error message
    by successfully padding 2D, 3D, 4D, and 5D tensors with mode='circular'.
    """
    # Dimensions explicitly claimed to be supported by the error message
    supported_dimensions = [2, 3, 4, 5]

    for dim in supported_dimensions:
        # Create a tensor of the target dimension (e.g., 4D -> (2, 2, 2, 2))
        shape = tuple([2] * dim)
        input_tensor = torch.empty(shape)
        
        # Define padding for the last dimension.
        # F.pad requires the padding tuple length to be 2 * input.dim().
        # We construct a tuple that pads the last dimension by 1 on each side,
        # and pads all other dimensions by 0.
        # The padding tuple format is (left, right, top, bottom, front, back, ...)
        # corresponding to the last dimension, second-to-last, etc.
        pad = tuple([1, 1] + [0] * (2 * (dim - 1)))

        try:
            # Attempt circular padding
            output_tensor = F.pad(input_tensor, pad, mode="circular")
            
            # Verify the output shape is correct
            expected_shape = list(shape)
            # sum(pad) is 2 (1+1 for the last dim, 0 for others)
            expected_shape[-1] += sum(pad)
            
            assert output_tensor.shape == tuple(expected_shape), \
                f"Shape mismatch for {dim}D input. Expected {expected_shape}, got {output_tensor.shape}"
            
            print(f" Test passed for {dim}D input: {input_tensor.shape} -> {output_tensor.shape}")

        except NotImplementedError as e:
            # If the bug is present, this block will execute.
            # We raise an AssertionError to explicitly flag the discrepancy
            # between the documented support (error message) and actual behavior.
            raise AssertionError(
                f"Bug detected for {dim}D input: {e}\n"
                "The error message claims support for this dimension, but the operation failed."
            )

if __name__ == "__main__":
    test_pad_circular_dimensions_consistency()