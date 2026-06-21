import torch
import pytest

# Helper function inspired by the pattern of tf.compat.v1.train.assert_global_step
# to validate tensor properties (type, dtype, shape) before the operation.
def assert_bmm_tensor_properties(tensor, expected_dtype, expected_ndims):
    """
    Asserts that the tensor is a torch.Tensor with the expected dtype and ndims.
    This mirrors the validation logic found in the similar API.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input must be a Tensor: {tensor}")
    
    if tensor.dtype != expected_dtype:
        raise TypeError(f"Input dtype mismatch. Expected {expected_dtype}, got {tensor.dtype}")
    
    if tensor.dim() != expected_ndims:
        raise TypeError(f"Input ndims mismatch. Expected {expected_ndims}, got {tensor.dim()}")

def test_torch_bmm_compile_out_dtype():
    """
    Test case for Issue #165892: torch.bmm + torch.compile with out_dtype.
    Verifies that torch.bmm handles the out_dtype argument correctly under torch.compile.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping test")

    # Setup inputs as per the bug report
    A = torch.rand((1, 1024, 1024), device="cuda", dtype=torch.float16)
    B = torch.rand((1, 1024, 1024), device="cuda", dtype=torch.float16)

    # Reuse the validation pattern from the similar API to ensure inputs are correct
    assert_bmm_tensor_properties(A, torch.float16, 3)
    assert_bmm_tensor_properties(B, torch.float16, 3)

    # The function from the bug report
    @torch.compile
    def linear(weight, input):
        return torch.bmm(input, weight, out_dtype=torch.float32)

    # Execute the function
    # Note: The bug report indicates this raises an AssertionError in Inductor:
    # "assert out_dtype is None, 'out_dtype is not supported for Triton'"
    # This test will fail if the bug is present, or pass if the fix is applied.
    try:
        result = linear(A, B)
        
        # If execution succeeds, verify the output dtype matches the request
        assert result.dtype == torch.float32, \
            f"Expected output dtype torch.float32, but got {result.dtype}"
            
    except AssertionError as e:
        if "out_dtype is not supported for Triton" in str(e):
            pytest.fail(f"Bug reproduced: Inductor does not support out_dtype for Triton. Error: {e}")
        else:
            raise