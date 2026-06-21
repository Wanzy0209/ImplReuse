import torch
import pytest

def test_lu_factor_singular_matrix_mps():
    """
    Test that torch.linalg.lu_factor raises a RuntimeError for singular matrices 
    on the MPS backend, consistent with CPU behavior.
    
    This test addresses the bug where MPS failed to raise an error for singular inputs.
    """
    # Skip if MPS is not available (e.g., on non-Mac hardware)
    if not torch.backends.mps.is_available():
        pytest.skip("MPS backend is not available")

    # Define a singular matrix (determinant is 0: 1*4 - 2*2 = 0)
    # This is the exact reproduction case from the issue.
    t = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device="mps")

    # Expect a RuntimeError because the matrix is singular.
    # The bug report indicates that on MPS this previously did not raise an error,
    # whereas on CPU it correctly did.
    with pytest.raises(RuntimeError):
        torch.linalg.lu_factor(t)

def test_lu_factor_singular_matrix_cpu():
    """
    Reference test to verify that the CPU backend correctly raises an error
    for singular matrices.
    """
    t = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device="cpu")

    with pytest.raises(RuntimeError):
        torch.linalg.lu_factor(t)