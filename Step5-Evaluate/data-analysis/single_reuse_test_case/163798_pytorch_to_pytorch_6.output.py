import torch

# Handle environments where torch.compile is not available (PyTorch < 2.0)
# by providing a mock pass-through decorator.
if not hasattr(torch, 'compile'):
    torch.compile = lambda **kwargs: lambda f: f

def test_lobpcg_compile():
    # Create a symmetric positive definite matrix for lobpcg
    A = torch.randn(5, 5)
    A = A @ A.T + torch.eye(5) * 0.1

    @torch.compile(fullgraph=False, backend="eager")
    def func(a):
        # Adapted call site: replacing a.tolist() with torch.lobpcg
        # lobpcg returns a tuple of (eigenvalues, eigenvectors)
        w, v = torch.lobpcg(a, k=1)
        
        # Adapted logic: using the extracted values in a calculation
        # w[0] is the largest eigenvalue
        return a * w[0]

    # Run the compiled function
    result = func(A)
    
    # Basic assertion to verify execution
    assert result.shape == A.shape
    assert torch.isfinite(result).all()

if __name__ == "__main__":
    test_lobpcg_compile()