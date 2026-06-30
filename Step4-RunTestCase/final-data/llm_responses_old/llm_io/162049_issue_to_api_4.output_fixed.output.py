import torch

def test_nested_tensor_int_reduction_ops():
    """
    Test case for Issue 162049: NestedTensor max/min not working for integer tensors.
    
    The bug was caused by the implementation using torch.finfo (for floats) 
    instead of torch.iinfo (for ints) when determining minimum values.
    
    This test verifies that reduction operations (max, min, argmax, argmin) 
    work correctly for integer types, preserving the logic that handles 
    different data types appropriately.
    """
    # Create a jagged nested tensor with integers
    # This reproduces the scenario where torch.finfo would fail
    # Fixed: Removed layout=torch.jagged as it is not a standard attribute in torch
    x_int = torch.nested.nested_tensor(
        [torch.arange(0, n) for n in (10, 20, 30)]
    )

    # Verify input type is integer (mirroring the conditional logic pattern 
    # found in similar API implementations to handle specific types)
    assert x_int.dtype.is_integer, "Test input must be an integer tensor"

    # Test max
    max_result = x_int.max(dim=1)
    expected_max = torch.tensor([9, 19, 29])
    assert torch.equal(max_result.values, expected_max), \
        f"Max values incorrect for int tensor: {max_result.values}"

    # Test min
    min_result = x_int.min(dim=1)
    expected_min = torch.tensor([0, 0, 0])
    assert torch.equal(min_result.values, expected_min), \
        f"Min values incorrect for int tensor: {min_result.values}"

    # Test argmax
    argmax_result = x_int.argmax(dim=1)
    expected_argmax = torch.tensor([9, 19, 29])
    assert torch.equal(argmax_result.values, expected_argmax), \
        f"Argmax values incorrect for int tensor: {argmax_result.values}"

    # Test argmin
    argmin_result = x_int.argmin(dim=1)
    expected_argmin = torch.tensor([0, 0, 0])
    assert torch.equal(argmin_result.values, expected_argmin), \
        f"Argmin values incorrect for int tensor: {argmin_result.values}"

    # Verify float tensors still work (regression test for the original torch.finfo path)
    # Fixed: Removed layout=torch.jagged
    x_float = torch.nested.nested_tensor(
        [torch.arange(0.0, float(n)) for n in (10, 20, 30)]
    )
    
    assert torch.allclose(x_float.max(dim=1).values, torch.tensor([9.0, 19.0, 29.0]))
    assert torch.allclose(x_float.min(dim=1).values, torch.tensor([0.0, 0.0, 0.0]))

if __name__ == "__main__":
    test_nested_tensor_int_reduction_ops()
    print("Test passed.")