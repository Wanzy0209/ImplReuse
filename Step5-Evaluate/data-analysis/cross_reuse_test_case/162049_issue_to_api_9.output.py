import torch
import sys

def test_nested_tensor_integer_reductions():
    """
    Test case for Issue #162049: NestedTensor max_dim / min_dim not working for integer tensors.
    
    The bug occurred because the implementation used torch.finfo (for floats) 
    instead of torch.iinfo (for integers) to determine minimum values.
    
    This test verifies that max, min, argmax, and argmin work correctly
    for integer datatypes (specifically int32 and int64) on jagged nested tensors.
    """
    
    # Create a nested jagged tensor with integer data (int64 by default)
    # This matches the minimal reproducer in the bug report
    # Note: Removed layout=torch.jagged as it is not a valid attribute in standard torch
    x_int64 = torch.nested.nested_tensor(
        [torch.arange(0, n) for n in (10, 20, 30)]
    )

    # Test max(dim=1)
    # Expected: tensor([9, 19, 29])
    max_result = x_int64.max(dim=1).values
    expected_max = torch.tensor([9, 19, 29])
    assert torch.equal(max_result, expected_max), f"Max failed for int64: {max_result} != {expected_max}"

    # Test min(dim=1)
    # Expected: tensor([0, 0, 0])
    min_result = x_int64.min(dim=1).values
    expected_min = torch.tensor([0, 0, 0])
    assert torch.equal(min_result, expected_min), f"Min failed for int64: {min_result} != {expected_min}"

    # Test argmax(dim=1)
    # Expected: tensor([9, 19, 29])
    argmax_result = x_int64.argmax(dim=1)
    expected_argmax = torch.tensor([9, 19, 29])
    assert torch.equal(argmax_result, expected_argmax), f"Argmax failed for int64: {argmax_result} != {expected_argmax}"

    # Test argmin(dim=1)
    # Expected: tensor([0, 0, 0])
    argmin_result = x_int64.argmin(dim=1)
    expected_argmin = torch.tensor([0, 0, 0])
    assert torch.equal(argmin_result, expected_argmin), f"Argmin failed for int64: {argmin_result} != {expected_argmin}"

    # Additional test for int32 to ensure robustness across integer types
    # Note: Removed layout=torch.jagged
    x_int32 = torch.nested.nested_tensor(
        [torch.arange(0, n, dtype=torch.int32) for n in (5, 15)]
    )

    assert torch.equal(x_int32.max(dim=1).values, torch.tensor([4, 14], dtype=torch.int32))
    assert torch.equal(x_int32.min(dim=1).values, torch.tensor([0, 0], dtype=torch.int32))

    print("All integer reduction tests passed.")

if __name__ == "__main__":
    test_nested_tensor_integer_reductions()