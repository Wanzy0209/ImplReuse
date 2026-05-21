import torch
import pytest

# Issue #164491: _scaled_mm and _int_mm are slow/error with row-major rhs matrix.
# The bug report indicates that _scaled_mm raises an error and _int_mm is slow
# when the right-hand side (rhs) matrix is in row-major format.
# 
# The similar API (tf.compat.v1.disable_v2_behavior) suggests a pattern of 
# verifying behavior changes based on specific configurations or states.
# Here, we adapt this pattern to verify behavior changes based on matrix layout (row-major vs column-major).

def test_internal_mm_layout_behavior():
    """
    Tests the behavior of internal PyTorch matrix multiplication functions 
    (_int_mm and _scaled_mm) with row-major right-hand side matrices.
    
    This test mirrors the logic of checking specific behavioral states 
    (similar to tf.compat.v1.disable_v2_behavior) but applied to tensor layouts.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, required for _scaled_mm/_int_mm")

    device = torch.device("cuda")
    M, K, N = 128, 64, 32

    # 1. Setup Inputs
    # Create LHS (M, K) and RHS (K, N)
    # Default PyTorch tensors are row-major (C-contiguous).
    lhs = torch.randn(M, K, dtype=torch.float16, device=device)
    rhs_row_major = torch.randn(K, N, dtype=torch.float16, device=device)
    
    # Verify RHS is row-major
    assert rhs_row_major.is_contiguous(), "Test setup failed: RHS should be row-major."

    # 2. Test torch._int_mm
    # Bug Description: "_int_mm does not [raise an error], but it is very slow... 
    # presumably does a transparent rhs.T.contiguous().T"
    # We test for correctness to ensure it doesn't crash, though performance is hard to assert here.
    if hasattr(torch, '_int_mm'):
        try:
            lhs_int8 = lhs.to(torch.int8)
            rhs_int8 = rhs_row_major.to(torch.int8)
            
            res_int_mm = torch._int_mm(lhs_int8, rhs_int8)
            
            # Verify correctness against standard matmul
            expected = lhs_int8.to(torch.float32) @ rhs_int8.to(torch.float32)
            assert torch.allclose(res_int_mm.to(torch.float32), expected), \
                "_int_mm produced incorrect results with row-major RHS."
        except Exception as e:
            pytest.fail(f"torch._int_mm failed unexpectedly with row-major RHS: {e}")
    else:
        pytest.skip("torch._int_mm is not available in this PyTorch version.")

    # 3. Test torch._scaled_mm
    # Bug Description: "_scaled_mm raises an error if the rhs matrix is row-major."
    # We expect this to potentially fail based on the bug report. 
    # If the bug is fixed, this should pass.
    if hasattr(torch, '_scaled_mm'):
        try:
            # _scaled_mm requires scales
            scale_a = torch.randn(M, 1, dtype=torch.float32, device=device)
            scale_b = torch.randn(1, N, dtype=torch.float32, device=device)
            
            # Attempt the operation
            res_scaled = torch._scaled_mm(lhs, rhs_row_major, scale_a, scale_b)
            
            # If we reach here, the bug might be fixed or the error condition wasn't met.
            # We check shape to ensure basic sanity.
            assert res_scaled.shape == (M, N), "_scaled_mm output shape mismatch."
            
        except RuntimeError as e:
            # If the error matches the bug description, we might want to assert it 
            # if we are testing for the bug's existence, or fail if we are testing for the fix.
            # Assuming we are testing for the fix (regression test), we fail here.
            # However, to preserve the bug reproduction logic, we acknowledge the error.
            pytest.fail(f"torch._scaled_mm raised an error with row-major RHS (Bug #164491): {e}")
    else:
        pytest.skip("torch._scaled_mm is not available in this PyTorch version.")

    # 4. Baseline: torch.matmul
    # Ensure standard matmul works fine with row-major (it should).
    try:
        res_matmul = torch.matmul(lhs, rhs_row_major)
        assert res_matmul.shape == (M, N)
    except Exception as e:
        pytest.fail(f"Standard torch.matmul failed with row-major RHS: {e}")

if __name__ == "__main__":
    test_internal_mm_layout_behavior()