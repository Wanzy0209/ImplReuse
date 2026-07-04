import torch
import pytest

def test_lu_factor_singular_matrix_mps():
    """
    Test that torch.linalg.lu_factor raises a RuntimeError on singular matrices
    when using the MPS backend, consistent with CPU behavior.
    
    This test addresses the issue where MPS backend failed to raise an error
    for singular matrices, potentially leading to division by zero in subsequent
    operations like lu_solve.
    """
    # Skip if MPS is not available
    if not torch.backends.mps.is_available():
        pytest.skip("MPS backend is not available")

    # Create a singular matrix (determinant is 0)
    # Matrix: [[1, 2], [2, 4]] -> 1*4 - 2*2 = 0
    t_mps = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device="mps")

    # Expect RuntimeError on singular matrix for MPS
    # The error message should match the CPU behavior regarding zero pivot
    with pytest.raises(RuntimeError, match="U\\[2,2\\] is zero"):
        torch.linalg.lu_factor(t_mps)

    # Verify CPU behavior matches to ensure consistency across backends
    t_cpu = t_mps.to("cpu")
    with pytest.raises(RuntimeError, match="U\\[2,2\\] is zero"):
        torch.linalg.lu_factor(t_cpu)