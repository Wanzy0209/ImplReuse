import torch
import torch._logging

# Mimic the logging setup from the bug report to observe recompiles
torch._logging.set_logs(recompiles=True)

def test_lobpcg_compile():
    # Setup device to match the original bug report context
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create a symmetric positive definite matrix
    # torch.lobpcg requires A to be symmetric
    n = 128
    k = 5
    X = torch.randn(n, n, dtype=torch.float32, device=device)
    A = X @ X.T

    # Define the function using the similar API: torch.lobpcg
    def solve_eigenvalues(A, k):
        # lobpcg finds the k largest eigenvalues/vectors
        return torch.lobpcg(A, k=k)

    # Compile the function, similar to pipe.transformer.compile_repeated_blocks()
    # This tests if torch.lobpcg interacts poorly with torch.compile (recompilation issues)
    compiled_solver = torch.compile(solve_eigenvalues)

    print("Running first inference (compilation)...")
    eigenvalues, eigenvectors = compiled_solver(A, k)

    print("Running second inference (checking for recompiles)...")
    eigenvalues_2, eigenvectors_2 = compiled_solver(A, k)

    # Assertions to verify correctness
    # Check if results are consistent between runs
    assert torch.allclose(eigenvalues, eigenvalues_2, atol=1e-4), "Eigenvalues differ between runs"
    assert torch.allclose(eigenvectors.abs(), eigenvectors_2.abs(), atol=1e-4), "Eigenvectors differ between runs"
    
    # Check basic property: A @ v = lambda * v
    # We check the first eigenpair
    lambda_0 = eigenvalues[0]
    v_0 = eigenvectors[:, 0]
    Av = A @ v_0
    lambda_v = lambda_0 * v_0
    assert torch.allclose(Av, lambda_v, atol=1e-3), "Eigenvector property A*v = lambda*v failed"

    print("Test passed successfully.")

if __name__ == "__main__":
    test_lobpcg_compile()