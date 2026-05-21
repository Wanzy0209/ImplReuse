import torch

def test_nested_tensor_integer_reductions():
    """
    Test that max, min, argmax, and argmin work correctly for integer NestedTensors.
    Regression test for Issue 162049 where torch.finfo was incorrectly used for integer types.
    """
    # Create a jagged nested tensor with integer datatype
    # Data: [0, 1, ..., 9], [0, 1, ..., 19], [0, 1, ..., 29]
    x = torch.nested.nested_tensor(
        [torch.arange(0, n) for n in (10, 20, 30)],
        layout=torch.jagged,
    )

    # Test max(dim=1)
    # Expected: [9, 19, 29]
    max_result = x.max(dim=1)
    assert torch.equal(max_result.values, torch.tensor([9, 19, 29])), \
        f"max(dim=1) failed. Expected [9, 19, 29], got {max_result.values}"

    # Test min(dim=1)
    # Expected: [0, 0, 0]
    min_result = x.min(dim=1)
    assert torch.equal(min_result.values, torch.tensor([0, 0, 0])), \
        f"min(dim=1) failed. Expected [0, 0, 0], got {min_result.values}"

    # Test argmax(dim=1)
    # Expected: [9, 19, 29]
    argmax_result = x.argmax(dim=1)
    assert torch.equal(argmax_result.values, torch.tensor([9, 19, 29])), \
        f"argmax(dim=1) failed. Expected [9, 19, 29], got {argmax_result.values}"

    # Test argmin(dim=1)
    # Expected: [0, 0, 0]
    argmin_result = x.argmin(dim=1)
    assert torch.equal(argmin_result.values, torch.tensor([0, 0, 0])), \
        f"argmin(dim=1) failed. Expected [0, 0, 0], got {argmin_result.values}"

    # Verify float types still work (ensure finfo path is not broken)
    x_float = torch.nested.nested_tensor(
        [torch.arange(0.0, float(n)) for n in (10, 20, 30)],
        layout=torch.jagged,
    )
    max_float = x_float.max(dim=1).values
    assert torch.allclose(max_float, torch.tensor([9.0, 19.0, 29.0])), \
        "Float max failed."

if __name__ == "__main__":
    test_nested_tensor_integer_reductions()
    print("All tests passed.")