import torch

def test_torch_normal_redundant_dtype_conversion():
    """
    Test case adapted from Issue 161611 for torch.normal.
    Verifies that calling .to(dtype) on a tensor that already has that dtype,
    without assigning the result, is redundant and ineffective.
    """
    # Setup: Define a query tensor to establish the target dtype and device
    query = torch.randn(10, dtype=torch.float32)
    L, S = 5, 5
    
    # Adaptation: Create a tensor using torch.normal.
    # Note: torch.normal does not accept dtype as a direct argument in the same way as zeros.
    # We explicitly cast it to query.dtype to simulate the "Already query.dtype" condition.
    attn_bias = torch.normal(0.0, 1.0, size=(L, S), device=query.device).to(query.dtype)
    
    # Verify initial condition: dtype is already query.dtype
    assert attn_bias.dtype == query.dtype, "Initial dtype should match query.dtype"
    
    # Perform an in-place operation (analogous to masked_fill_ in the bug report)
    # This operation does not change the dtype.
    attn_bias.fill_(0.0)
    
    # The redundant call from the bug report:
    # attn_bias.to(query.dtype)
    # Since the result is not assigned, this line should have no effect on attn_bias.
    attn_bias.to(query.dtype)
    
    # Verify that attn_bias is still float32 (query.dtype)
    assert attn_bias.dtype == query.dtype, "Dtype should remain query.dtype after redundant .to()"
    
    # Verify that the content is still 0.0 (from fill_), 
    # confirming the unassigned .to() did not replace the tensor object.
    assert torch.all(attn_bias == 0.0), "Tensor content should remain unchanged"
    
    print("Test passed: Redundant .to() call was ineffective as expected.")

if __name__ == "__main__":
    test_torch_normal_redundant_dtype_conversion()