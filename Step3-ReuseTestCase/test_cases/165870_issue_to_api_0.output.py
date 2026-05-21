import torch
import pytest

def test_lu_factor_singular_matrix_consistency():
    """
    Test that torch.linalg.lu_factor raises a RuntimeError on singular matrices
    consistently across CPU and MPS backends.
    
    This test mirrors the context-checking pattern seen in the similar API
    (checking execution context/device availability) to ensure behavior is 
    validated where applicable.
    """
    # Define a singular matrix (determinant is 0)
    # 1*4 - 2*2 = 0
    t = torch.tensor([[1.0, 2.0], [2.0, 4.0]])

    # Expected error message substring
    error_match = "U\\[2,2\\] is zero"

    # Test on CPU
    with pytest.raises(RuntimeError, match=error_match):
        torch.linalg.lu_factor(t)

    # Test on MPS if available
    # This mirrors the pattern of checking context (e.g., executing_eagerly)
    # before performing the specific operation.
    if torch.backends.mps.is_available():
        t_mps = t.to("mps")
        with pytest.raises(RuntimeError, match=error_match):
            torch.linalg.lu_factor(t_mps)