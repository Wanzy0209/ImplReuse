import torch
import pytest

def test_mm_out_dtype_with_is_complex():
    """
    Test case for Issue 163275: torch compile does not handle out_dtype argument in torch.mm.
    This test preserves the original bug reproduction logic (torch.mm with out_dtype inside torch.compile)
    and leverages the similar API torch.is_complex to verify the output properties.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    A = torch.rand((1024, 1024), device="cuda", dtype=torch.float16)
    B = torch.rand((1024, 1024), device="cuda", dtype=torch.float16)

    @torch.compile
    def linear_check(weight, input):
        # Original bug reproduction logic: torch.mm with out_dtype
        result = torch.mm(input, weight, out_dtype=torch.float32)
        
        # Leveraging similar API: torch.is_complex
        # We check if the result is complex (it should be False for float32)
        is_complex_result = torch.is_complex(result)
        
        return result, is_complex_result

    result, is_complex_val = linear_check(A, B)

    # Assertions to verify correctness
    assert result.dtype == torch.float32, f"Expected float32, got {result.dtype}"
    assert is_complex_val == False, "Expected result to not be complex"