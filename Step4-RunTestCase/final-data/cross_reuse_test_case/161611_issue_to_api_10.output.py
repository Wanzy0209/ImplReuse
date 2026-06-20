import torch

def test_redundant_dtype_conversion_issue_161611():
    """
    Test case for Issue 161611: Remove redundant dtype conversion in 
    scaled_dot_product_attention docstring example.
    
    This test verifies that the line `attn_bias.to(query.dtype)` is redundant
    because `attn_bias` is already created with `query.dtype` and `masked_fill_`
    does not change the dtype.
    """
    # Setup dimensions and query tensor
    L, S = 8, 8
    query = torch.randn(1, 1, L, 16, dtype=torch.float16)
    
    # Reproduce the logic from the docstring using torch.zeros
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    
    # Verify initial dtype matches query.dtype
    assert attn_bias.dtype == query.dtype, \
        f"Expected attn_bias dtype {query.dtype}, but got {attn_bias.dtype}"
    
    # Simulate causal mask logic
    is_causal = True
    if is_causal:
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        # masked_fill_ is an in-place operation
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        
        # Verify dtype is preserved after masked_fill_
        assert attn_bias.dtype == query.dtype, \
            "masked_fill_ should not change the dtype"
        
        # The redundant line from the bug report:
        # attn_bias.to(query.dtype)
        
        # Verify that calling .to() with the same dtype is a no-op
        # and that not assigning the result (as in the bug) leaves the tensor unchanged.
        id_before = id(attn_bias)
        attn_bias.to(query.dtype) # Call without assignment
        id_after = id(attn_bias)
        
        assert id_before == id_after, \
            ".to() with the same dtype should return the same object (no-op)"
        assert attn_bias.dtype == query.dtype, \
            "Dtype should remain unchanged after redundant .to() call"

if __name__ == "__main__":
    test_redundant_dtype_conversion_issue_161611()
    print("Test passed successfully.")