import torch

def test_nested_tensor_integer_reductions():
    """
    Test case for Issue 162049: NestedTensor max_dim / min_dim not working for integer tensors.
    
    This test leverages the jagged data structure pattern found in the similar API 
    (tf.feature_column.crossed_column) to ensure the reduction operations handle 
    integer datatypes correctly (using torch.iinfo instead of torch.finfo).
    """
    # Create jagged integer data, mimicking the sparse structure of the crossed_column example.
    # Explicitly using int32 to ensure type-specific handling is tested.
    data = [
        torch.tensor([10, 20], dtype=torch.int32),   # Represents first feature set
        torch.tensor([30, 40, 50], dtype=torch.int32), # Represents second feature set
        torch.tensor([60], dtype=torch.int32)        # Represents third feature set
    ]

    # Construct the NestedTensor with jagged layout
    x = torch.nested.nested_tensor(data, layout=torch.jagged)

    # Test max(dim=...) - Original failing operation
    max_result = x.max(dim=1)
    expected_max_values = torch.tensor([20, 50, 60], dtype=torch.int32)
    assert torch.equal(max_result.values, expected_max_values), \
        f"Max values mismatch: {max_result.values} != {expected_max_values}"

    # Test min(dim=...) - Also mentioned in the bug report
    min_result = x.min(dim=1)
    expected_min_values = torch.tensor([10, 30, 60], dtype=torch.int32)
    assert torch.equal(min_result.values, expected_min_values), \
        f"Min values mismatch: {min_result.values} != {expected_min_values}"

    # Test argmax(dim=...)
    argmax_result = x.argmax(dim=1)
    expected_argmax = torch.tensor([1, 2, 0], dtype=torch.long)
    assert torch.equal(argmax_result, expected_argmax), \
        f"Argmax mismatch: {argmax_result} != {expected_argmax}"

    # Test argmin(dim=...)
    argmin_result = x.argmin(dim=1)
    expected_argmin = torch.tensor([0, 0, 0], dtype=torch.long)
    assert torch.equal(argmin_result, expected_argmin), \
        f"Argmin mismatch: {argmin_result} != {expected_argmin}"

    print("All integer reduction tests passed for NestedTensor.")

if __name__ == "__main__":
    test_nested_tensor_integer_reductions()