import torch

def test_matrix_power_redundant_dtype_conversion():
    """
    Test case adapted from Issue 161611 to verify redundant dtype conversion behavior
    with torch.linalg.matrix_power (similar to torch.zeros context in the original bug).
    
    The original bug highlighted that calling tensor.to(original_dtype) without assignment
    is redundant and ineffective. This test verifies that matrix_power preserves the dtype
    such that a subsequent redundant conversion is indeed unnecessary.
    """
    # Setup: Create a matrix with a specific dtype (float32)
    original_dtype = torch.float32
    input_matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=original_dtype)

    # Action: Compute matrix power
    # For floating point inputs, matrix_power generally preserves the input dtype
    result = torch.linalg.matrix_power(input_matrix, 2)

    # Verification 1: Check that the result already has the target dtype
    assert result.dtype == original_dtype, \
        f"Expected dtype {original_dtype}, but got {result.dtype}"

    # Capture state before the redundant operation
    result_clone = result.clone()

    # Action: Perform the redundant conversion (mimicking the bug report's pattern)
    # Issue: Calling .to() on the same dtype without assigning the result is ineffective.
    result.to(original_dtype)

    # Verification 2: Ensure the tensor 'result' is unchanged
    # This confirms that the line 'result.to(original_dtype)' can be safely removed.
    assert result.dtype == original_dtype, "Dtype should remain unchanged"
    assert torch.equal(result, result_clone), "Data should remain unchanged"

    print("Test passed: Redundant dtype conversion is verified as ineffective.")

if __name__ == "__main__":
    test_matrix_power_redundant_dtype_conversion()