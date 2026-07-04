import torch

def test_torch_eye_redundant_dtype_conversion():
    """
    Test case adapted from Issue 161611 to verify torch.eye behavior.
    
    This test verifies that torch.eye respects the dtype argument upon creation
    and that in-place operations (like masked_fill_) preserve the dtype, 
    making subsequent .to() calls redundant.
    """
    # Setup variables similar to the original bug report context
    N = 4
    query = torch.randn(N, N, dtype=torch.float16)
    
    # Adaptation: Use torch.eye instead of torch.zeros
    # Original: attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    eye_matrix = torch.eye(N, dtype=query.dtype, device=query.device)
    
    # Verify 1: The tensor is created with the correct dtype immediately
    assert eye_matrix.dtype == query.dtype, "torch.eye did not respect the specified dtype"
    
    # Verify 2: Perform an in-place operation (similar to masked_fill_ in the bug)
    # masked_fill_ is an in-place operation that does not change the tensor's dtype
    mask = torch.ones(N, N, dtype=torch.bool)
    eye_matrix.masked_fill_(mask, 0.0)
    
    # Verify 3: The dtype remains correct after the in-place operation
    assert eye_matrix.dtype == query.dtype, "Dtype changed unexpectedly after in-place operation"
    
    # Conclusion: A call like `eye_matrix.to(query.dtype)` here would be redundant
    # because the dtype is already query.dtype.
    assert eye_matrix.dtype == torch.float16

if __name__ == "__main__":
    test_torch_eye_redundant_dtype_conversion()
    print("Test passed.")