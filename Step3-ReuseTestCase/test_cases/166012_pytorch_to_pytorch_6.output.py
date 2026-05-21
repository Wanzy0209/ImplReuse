import torch
import unittest

class TestLobpcgCompileConsistency(unittest.TestCase):
    def test_lobpcg_compile_cache_hit_miss(self):
        """
        Test that torch.lobpcg behaves consistently when run through torch.compile
        on both cache miss and cache hit.
        
        This test is derived from Issue 166012 which highlights inconsistencies
        in compilation artifacts (tlparse entries) between cache hits and misses.
        We verify that the numerical results remain consistent.
        """
        # Setup: Create a symmetric positive definite matrix
        torch.manual_seed(42)
        A = torch.randn(10, 10, dtype=torch.float32)
        A = A @ A.T + torch.eye(10, dtype=torch.float32)  # Make it SPD

        # Define the function using the similar API: torch.lobpcg
        def lobpcg_func(A_tensor):
            k = 2
            # Initial guess
            X = torch.randn(A_tensor.shape[0], k, dtype=A_tensor.dtype)
            # Call torch.lobpcg
            eigenvalues, eigenvectors = torch.lobpcg(A_tensor, k=k, X=X)
            return eigenvalues, eigenvectors

        # Compile the function (Original API Under Test context)
        compiled_lobpcg = torch.compile(lobpcg_func, mode="reduce-overhead")

        # Run 1: Cache Miss
        evals_miss, evecs_miss = compiled_lobpcg(A)

        # Run 2: Cache Hit
        evals_hit, evecs_hit = compiled_lobpcg(A)

        # Assertions: Check consistency between cache miss and hit
        # Note: Eigenvectors might differ by sign, so we check absolute values or norms
        self.assertTrue(
            torch.allclose(evals_miss, evals_hit, atol=1e-4),
            "Eigenvalues differ between cache miss and hit"
        )
        
        # For eigenvectors, check if they span the same subspace or are close
        # (allowing for sign flips)
        self.assertTrue(
            torch.allclose(torch.abs(evecs_miss), torch.abs(evecs_hit), atol=1e-4),
            "Eigenvectors differ significantly between cache miss and hit"
        )

        # Optional: Verify against eager mode to ensure compilation didn't break logic
        evals_eager, evecs_eager = lobpcg_func(A)
        self.assertTrue(
            torch.allclose(evals_miss, evals_eager, atol=1e-4),
            "Compiled eigenvalues differ from eager mode"
        )

if __name__ == "__main__":
    unittest.main()