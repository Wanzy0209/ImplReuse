import torch

def test_lobpcg_compile():
    """
    Test case to verify torch.lobpcg works correctly with torch.compile.
    This test is derived from the context of Issue #161372 (torch.compile regression),
    ensuring that similar APIs (torch.lobpcg) remain functional under compilation.
    """
    torch.manual_seed(42)

    # Define a function that uses torch.lobpcg
    def eigen_solver(A, k):
        # torch.lobpcg finds the k largest eigenvalues and corresponding eigenvectors
        # of a symmetric positive definite generalized eigenvalue problem.
        return torch.lobpcg(A, k=k)

    # Compile the function using torch.compile
    compiled_solver = torch.compile(eigen_solver)

    # Setup test data: Create a symmetric positive definite matrix
    n = 16
    k = 3
    X = torch.randn(n, n, dtype=torch.float64)
    # A = X @ X.T + epsilon * I ensures positive definiteness
    A = X @ X.T + 1e-3 * torch.eye(n, dtype=torch.float64)

    # Run the eager version
    w_eager, v_eager = eigen_solver(A, k)

    # Run the compiled version
    # This checks if torch.compile handles the lobpcg operations without hitting
    # recompilation limits or errors similar to the reported bug.
    w_compiled, v_compiled = compiled_solver(A, k)

    # Verify results
    # Eigenvalues should match closely
    assert torch.allclose(w_eager, w_compiled, atol=1e-5), "Eigenvalues mismatch between eager and compiled"
    
    # Eigenvectors can differ by sign, so we check absolute values or align signs
    # Here we check the residual norm: ||A*v - lambda*v|| should be small for both
    residual_eager = torch.norm(A @ v_eager - w_eager * v_eager)
    residual_compiled = torch.norm(A @ v_compiled - w_compiled * v_compiled)
    
    assert residual_eager < 1e-5, "Eager mode residual too high"
    assert residual_compiled < 1e-5, "Compiled mode residual too high"

    print("Test passed: torch.lobpcg is compatible with torch.compile")

if __name__ == "__main__":
    test_lobpcg_compile()