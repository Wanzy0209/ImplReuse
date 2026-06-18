import torch

def test_dist_redundant_dtype_conversion():
    """
    Test case adapted from Issue 161611.
    Verifies that torch.dist handles dtypes correctly and that
    redundant .to() calls (as seen in the bug) are indeed no-ops.
    """
    # Setup: Define a specific dtype
    target_dtype = torch.float32

    # Adaptation: Use torch.zeros (original API) to create inputs for torch.dist (similar API)
    # In the bug, torch.zeros was used to create attn_bias.
    # Here we use it to create inputs for distance calculation.
    input_a = torch.zeros(5, dtype=target_dtype)
    input_b = torch.ones(5, dtype=target_dtype)

    # Call the Similar API
    result = torch.dist(input_a, input_b)

    # Check 1: Verify the result has the expected dtype
    assert result.dtype == target_dtype, \
        f"torch.dist returned {result.dtype}, expected {target_dtype}"

    # Check 2: Mimic the bug scenario - redundant conversion
    # Bug: attn_bias.to(query.dtype) was redundant and unassigned.
    # We verify that calling .to() on the result with its own dtype is redundant.
    redundant_result = result.to(target_dtype)

    # The values should be identical
    assert torch.equal(result, redundant_result), \
        "Redundant .to() call changed the tensor value unexpectedly"

    # The dtypes should be identical
    assert redundant_result.dtype == target_dtype, \
        "Redundant .to() call changed the dtype unexpectedly"

    # Check 3: Verify the unassigned call (the specific bug pattern) does nothing
    # In the bug: attn_bias.to(query.dtype) # result not assigned
    # We verify the tensor object remains the same (or value is same)
    original_value = result.item()
    result.to(target_dtype) # Unassigned, like the bug
    assert result.item() == original_value, \
        "Unassigned .to() call modified the tensor in-place unexpectedly"

if __name__ == "__main__":
    test_dist_redundant_dtype_conversion()
    print("Test passed successfully.")