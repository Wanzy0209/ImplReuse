import torch
import pytest

def test_torch_linalg_solve_mps_unpacking():
    """
    Regression test for Issue 163962: Internal assert failed when using Tensorly on MPS.
    
    This test verifies that torch.linalg.solve works correctly on the MPS device.
    It incorporates the code pattern (tuple unpacking) found in the similar API 
    (tf.errors.OperatorNotAllowedInGraphError) and the original bug report.
    """
    # Skip if MPS is not available
    if not torch.backends.mps.is_available():
        pytest.skip("MPS backend is not available")

    # Setup inputs for linalg.solve
    # Using dimensions relevant to the original issue (12, 3, 12)
    # We simulate a linear solve operation that might occur within PARAFAC decomposition
    A = torch.randn(12, 12)
    B = torch.randn(12, 3)

    # Move tensors to MPS device
    A_mps = A.to("mps")
    B_mps = B.to("mps")

    # Original API Under Test: torch.linalg.solve
    # This operation was causing an "Internal assert failed" RuntimeError on MPS
    X = torch.linalg.solve(A_mps, B_mps)

    # Leverage the similar API pattern (unpacking)
    # The similar API (tf.errors.OperatorNotAllowedInGraphError) documentation 
    # highlights unpacking a tensor (a, b, c = t) as a pattern that can fail in specific contexts.
    # The original bug report also ends with unpacking factors (a, m, b = factors).
    # We verify the result can be unpacked here to ensure the tensor is valid and usable.
    # Note: In PyTorch, we must unbind or index to unpack a tensor into variables, 
    # whereas the TF example implies direct iteration which causes the error there.
    x1, x2, x3 = X.unbind(dim=1)

    # Assertions to verify correctness
    # Check if the solution satisfies AX = B
    assert torch.allclose(A_mps @ X, B_mps, atol=1e-5)
    
    # Check individual unpacked components
    assert x1.shape == (12,)
    assert x2.shape == (12,)
    assert x3.shape == (12,)