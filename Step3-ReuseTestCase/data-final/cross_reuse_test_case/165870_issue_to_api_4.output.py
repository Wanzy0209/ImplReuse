import torch
import pytest

def test_lu_factor_singular_matrix_mps():
    """
    Test that torch.linalg.lu_factor raises a RuntimeError for singular matrices
    on the MPS device, matching the CPU behavior.
    
    This test addresses the issue where MPS backend silently failed to raise
    an error for singular inputs, unlike the CPU backend.
    """
    # Skip if MPS is not available (e.g., running on Linux or Windows)
    if not torch.backends.mps.is_available():
        pytest.skip("MPS backend is not available")

    # Create a singular matrix (determinant is 0)
    # 1*4 - 2*2 = 0
    t = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device="mps")

    # The expected behavior is to raise a RuntimeError regarding the zero pivot.
    # This matches the CPU behavior described in the bug report.
    with pytest.raises(RuntimeError, match="U\\[2,2\\] is zero"):
        torch.linalg.lu_factor(t)