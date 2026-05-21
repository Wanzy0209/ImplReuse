import torch
import unittest
import sys

# The original bug report describes a timeout issue when using torch.compile
# with distributed tensors (DTensor). This test case adapts that scenario
# to the similar API torch.lobpcg to verify its stability and correctness
# when used with torch.compile.

class TestLobpcgCompile(unittest.TestCase):
    def test_lobpcg_compile_basic(self):
        """
        Test that torch.lobpcg works correctly when compiled with torch.compile.
        This mirrors the context of the original failing test (torch.compile + complex op).
        """
        # Setup: Create a symmetric positive definite matrix
        n = 16  # Matrix size
        k = 3   # Number of eigenvalues to compute
        dtype = torch.float32
        
        # Generate a random matrix A
        A = torch.randn(n, n, dtype=dtype)
        # Make A symmetric positive definite: A = A @ A.T + n*I
        A = A @ A.T + n * torch.eye(n, dtype=dtype)
        
        # Initial guess for eigenvectors
        X = torch.randn(n, k, dtype=dtype)

        # Define the function using the similar API (torch.lobpcg)
        def run_lobpcg(A, X):
            # lobpcg returns a tuple of (eigenvalues, eigenvectors)
            return torch.lobpcg(A, k=k, X=X)

        # Compile the function using torch.compile
        # This is the critical part related to the original bug report
        compiled_run_lobpcg = torch.compile(run_lobpcg)

        # Execute the compiled function
        try:
            eigenvalues, eigenvectors = compiled_run_lobpcg(A, X)
            
            # Verification 1: Check output shapes
            self.assertEqual(eigenvalues.shape[0], k)
            self.assertEqual(eigenvectors.shape, (n, k))
            
            # Verification 2: Check the eigenvalue equation A @ v = lambda * v
            # Calculate residual
            Av = torch.matmul(A, eigenvectors)
            lambda_v = eigenvectors * eigenvalues.view(1, -1)
            residual = torch.norm(Av - lambda_v)
            
            # Assert that the residual is small (solution is correct)
            self.assertLess(residual, 1e-2, "LOBPCG failed to converge or produced incorrect results")
            
        except Exception as e:
            self.fail(f"torch.compile(torch.lobpcg) raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()