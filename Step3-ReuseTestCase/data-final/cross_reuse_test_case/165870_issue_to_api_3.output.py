import torch
import pytest

def test_lu_factor_singular_matrix():
    """
    Test that torch.linalg.lu_factor raises a RuntimeError for singular matrices
    on both CPU and MPS devices.
    
    This test addresses the issue where MPS backend did not raise an error
    for singular matrices, unlike the CPU backend.
    """
    # Create a singular matrix (determinant is 0)
    # 1*4 - 2*2 = 0
    singular_matrix = torch.tensor([[1.0, 2.0], [2.0, 4.0]])

    # Test on CPU
    with pytest.raises(RuntimeError, match="U\\[2,2\\] is zero"):
        torch.linalg.lu_factor(singular_matrix)

    # Test on MPS if available
    if torch.backends.mps.is_available():
        mps_matrix = singular_matrix.to("mps")
        with pytest.raises(RuntimeError, match="U\\[2,2\\] is zero"):
            torch.linalg.lu_factor(mps_matrix)