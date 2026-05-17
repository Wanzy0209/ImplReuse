import torch
import pytest

def test_compile_isnan_dynamic_shape_regression():
    """
    Regression test for Issue #161372: torch.compile regression in 2.8.0.
    
    The issue reported a recompilation limit error due to tensor size mismatches
    (expected 77, actual 78) in an LLM implementation. This test verifies that
    torch.compile handles dynamic shapes correctly when using torch.isnan
    (the similar API identified) and torch.where (seen in the error stack trace).
    """
    
    def forward_pass(x):
        # Use torch.isnan (Similar API) to generate a mask.
        # The implementation of isnan uses full_like/make_pointwise which
        # interacts with the tensor shape.
        mask = torch.isnan(x)
        
        # Use torch.where (from the original error log) to apply the mask.
        # This pattern is common in LLMs for masking attention or loss values.
        result = torch.where(mask, torch.zeros_like(x), x)
        return result

    # Compile the function
    compiled_fn = torch.compile(forward_pass)

    # Test Case 1: Sequence length 77 (from bug report)
    # Shape: (Batch, Seq_Len, Hidden_Dim)
    input_77 = torch.randn(2, 77, 128)
    output_77 = compiled_fn(input_77)
    
    # Verify output shape matches input
    assert output_77.shape == input_77.shape
    # Verify no NaNs remain (sanity check logic)
    assert not torch.any(torch.isnan(output_77))

    # Test Case 2: Sequence length 78 (from bug report)
    # This change in size triggered the "size mismatch" and recompile limit error.
    input_78 = torch.randn(2, 78, 128)
    output_78 = compiled_fn(input_78)
    
    # Verify output shape matches input
    assert output_78.shape == input_78.shape
    assert not torch.any(torch.isnan(output_78))

    # If the function completes without hitting torch._dynamo.config.recompile_limit,
    # the regression is likely fixed/passed.

if __name__ == "__main__":
    test_compile_isnan_dynamic_shape_regression()
    print("Test passed successfully.")