import torch
import pytest

# Reusing the code pattern from the similar API (tf.autograph.trace)
# to log execution details, which aligns with the debugging nature of the issue.
def trace(*args):
    """Helper to trace execution, mimicking the provided similar API."""
    print(*args)

def test_torch_linalg_solve_mps():
    """
    Test case for Issue 163962: Internal assert failed when using Tensorly on MPS.
    
    This test isolates the underlying API call (torch.linalg.solve) that likely 
    causes the failure in the PARAFAC decomposition on the MPS device.
    """
    # Skip if MPS is not available (e.g., on non-Apple Silicon or older macOS)
    if not torch.backends.mps.is_available():
        pytest.skip("MPS backend is not available")

    # Reproduce logic from the issue: creating tensors and moving to MPS
    # The issue uses a tensor of shape (12, 3, 12) and rank 12.
    # We create a square matrix for linalg.solve based on the rank dimension.
    rank = 12
    
    # Create a random matrix A and vector b on MPS
    # We ensure A is positive definite to guarantee a solution exists
    A = torch.randn(rank, rank, device="mps")
    A = A @ A.T + 1e-3 * torch.eye(rank, device="mps")
    b = torch.randn(rank, device="mps")

    # Use the trace function to log inputs, similar to the issue's print(x.shape)
    trace("Input A shape:", A.shape, "device:", A.device)
    trace("Input b shape:", b.shape, "device:", b.device)

    try:
        # This is the Original API Under Test
        # The bug report suggests this fails with "Internal assert failed" on MPS
        x = torch.linalg.solve(A, b)
        
        trace("Output x shape:", x.shape)
        
        # Assertions to verify correctness if it doesn't crash
        assert x.shape == b.shape
        assert x.device == A.device
        
        # Verify the solution
        residual = torch.norm(A @ x - b)
        trace("Residual norm:", residual.item())
        assert residual < 1e-4

    except RuntimeError as e:
        # Catch the specific error mentioned in the bug report
        if "Internal assert failed" in str(e):
            trace("Error caught:", str(e))
            pytest.fail(f"torch.linalg.solve failed on MPS with Internal assert: {e}")
        else:
            raise

if __name__ == "__main__":
    test_torch_linalg_solve_mps()