"""
Minimal reproducible example for torch.lobpcg
"""

import torch

def main() -> None:
    # Setup: Create a symmetric positive definite matrix
    n = 20
    k = 5
    torch.manual_seed(42)
    
    # Generate a random matrix and make it symmetric positive definite
    A = torch.randn(n, n, dtype=torch.float64)
    A = A @ A.T + 1e-3 * torch.eye(n)

    # Initial guess for eigenvectors
    X = torch.randn(n, k, dtype=torch.float64)

    # Call torch.lobpcg
    # This is the API under test
    eigenvalues, eigenvectors = torch.lobpcg(A, k=k, X=X)

    # Verification
    # 1. Check shapes
    assert eigenvalues.shape == (k,), f"Expected eigenvalues shape ({k},), got {eigenvalues.shape}"
    assert eigenvectors.shape == (n, k), f"Expected eigenvectors shape ({n}, {k}), got {eigenvectors.shape}"

    # 2. Verify eigenvalues are real (lobpcg returns real for symmetric matrices)
    assert eigenvalues.is_floating_point()

    # 3. Verify orthogonality: V^T @ V should be Identity
    # Note: lobpcg returns normalized eigenvectors
    ortho_matrix = eigenvectors.T @ eigenvectors
    identity = torch.eye(k, dtype=torch.float64)
    assert torch.allclose(ortho_matrix, identity, atol=1e-6), "Eigenvectors are not orthogonal"

    # 4. Verify eigenvalue equation: A @ v = lambda * v
    # We check the residual for the largest eigenvalue
    idx_max = torch.argmax(eigenvalues)
    v_max = eigenvectors[:, idx_max]
    lambda_max = eigenvalues[idx_max]
    Av = A @ v_max
    lambda_v = lambda_max * v_max
    residual = torch.norm(Av - lambda_v)
    assert residual < 1e-5, f"Residual {residual} is too high for eigenvalue equation"

    print("torch.lobpcg test passed successfully.")

if __name__ == "__main__":
    main()