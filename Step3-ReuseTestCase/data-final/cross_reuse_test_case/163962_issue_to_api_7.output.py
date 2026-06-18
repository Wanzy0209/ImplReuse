import torch
import sys

# Leveraging the similar API pattern (tf.profiler.experimental.client.trace)
# to instrument the test case for debugging purposes.
def trace(*args):
    """Traces argument information at runtime."""
    print("[TRACE]", *args)

def test_torch_linalg_solve_mps_assert():
    """
    Test case for Issue 163962: Internal assert failed when using Tensorly on MPS.
    The root cause is identified as torch.linalg.solve failing on MPS.
    This test reproduces the logic using the core API directly.
    """
    if not torch.backends.mps.is_available():
        trace("MPS backend is not available. Skipping test.")
        return

    trace("Initializing test on MPS device.")
    
    # Dimensions from the original bug report: tensor shape (12, 3, 12), rank 12
    rank = 12
    mode_dim = 12 
    
    # Setup inputs for torch.linalg.solve
    # In PARAFAC, this often involves solving a system where the matrix is 
    # related to the Khatri-Rao product (Rank x Rank).
    # We create a random positive definite matrix to ensure solvability.
    trace(f"Creating matrices with Rank={rank}, Mode_Dim={mode_dim}")
    
    A = torch.randn(rank, rank, device="mps")
    A = A @ A.T + torch.eye(rank, device="mps") * 1e-3 # Ensure invertibility
    B = torch.randn(mode_dim, rank, device="mps")

    trace("Input A shape:", A.shape, "dtype:", A.dtype)
    trace("Input B shape:", B.shape, "dtype:", B.dtype)

    try:
        # This is the Original API Under Test that triggers the assert
        trace("Calling torch.linalg.solve...")
        X = torch.linalg.solve(A, B)
        
        trace("Operation successful. Result shape:", X.shape)
        
        # Basic assertion to ensure computation happened
        assert X.shape == (mode_dim, rank)
        assert torch.allclose(A @ X, B, atol=1e-4)
        
    except RuntimeError as e:
        trace("RuntimeError caught:", str(e))
        if "Internal assert" in str(e):
            print("Bug reproduced: Internal assert failed on MPS.")
            raise
        else:
            raise

if __name__ == "__main__":
    test_torch_linalg_solve_mps_assert()