import torch

def test_torch_norm_dtype_redundancy():
    """
    Test case adapted from Issue 161611 to verify torch.norm behavior regarding dtype conversions.
    The original issue highlighted a redundant .to() call in a docstring example for scaled_dot_product_attention.
    This test verifies that torch.norm handles dtypes correctly, making similar redundant conversions unnecessary.
    """
    # Setup mimicking the bug report context
    query = torch.randn(10, 10, dtype=torch.float32)
    L, S = query.shape

    # Original code: attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    # Adapted code: We use torch.norm to generate a tensor.
    # torch.norm on float32 input returns float32 output.
    norm_tensor = torch.norm(query, dim=1, keepdim=True)

    # Verify the dtype is already correct (matches query.dtype)
    assert norm_tensor.dtype == query.dtype, \
        f"Expected dtype {query.dtype}, but got {norm_tensor.dtype}"

    # Simulate an in-place operation similar to the bug report's masked_fill_
    # (e.g., preventing division by zero)
    norm_tensor.clamp_(min=1e-6)

    # The bug report highlights a redundant .to() call that is not assigned.
    # We verify that calling .to(query.dtype) is redundant here.
    # 1. Check if the operation is a no-op (returns self)
    converted = norm_tensor.to(query.dtype)
    assert converted is norm_tensor, ".to() should return self if dtype matches"

    # 2. Verify the unassigned call (the specific bug pattern) does not change the tensor
    norm_tensor.to(query.dtype)
    assert norm_tensor.dtype == query.dtype

    # Additionally, verify the similar API: torch._numpy.linalg.norm
    # which has specific dtype handling logic (_atleast_float_1).
    from torch._numpy.linalg import norm as np_norm

    # Case 1: Float input (already correct type)
    x_float = torch.randn(2, 2, dtype=torch.float32)
    y_float = np_norm(x_float)
    # _atleast_float_1 keeps it float.
    assert y_float.dtype == torch.float32
    # Redundant conversion check
    y_float.to(torch.float32) # Should be no-op

    # Case 2: Int input (requires conversion)
    x_int = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    y_int = np_norm(x_int)
    # _atleast_float_1 converts int to float.
    assert y_int.dtype.is_floating_point
    # If we wanted to ensure it's the default float, it already is.
    # Redundant conversion to its own type
    y_int.to(y_int.dtype)

if __name__ == "__main__":
    test_torch_norm_dtype_redundancy()
    print("Test passed.")