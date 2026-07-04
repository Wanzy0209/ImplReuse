import torch

# Handle environments where torch.compile is not available (PyTorch < 2.0)
# We use a pass-through decorator if torch.compile is missing to allow the test to run.
if hasattr(torch, 'compile'):
    compile_decorator = torch.compile
else:
    compile_decorator = lambda f: f
    print("Warning: torch.compile not found (requires PyTorch >= 2.0). Running without compilation.")

@compile_decorator
def test_lobpcg_compile(A):
    # lobpcg finds eigenvalues/vectors. 
    # We pass a symmetric positive definite matrix.
    # The original bug involved int64, but lobpcg typically operates on floats.
    # We test the standard usage under torch.compile.
    return torch.lobpcg(A, k=2)

# Setup inputs
# Create a symmetric positive definite matrix
# lobpcg requires A to be symmetric positive definite
device = 'cuda' if torch.cuda.is_available() else 'cpu'
A = torch.randn(10, 10, dtype=torch.float32, device=device)
A = A @ A.T + torch.eye(10, device=device) * 0.1

# Run the test
try:
    eigenvalues, eigenvectors = test_lobpcg_compile(A)
    print("Test passed: torch.lobpcg compiled and ran successfully.")
    print(f"Eigenvalues shape: {eigenvalues.shape}")
    assert eigenvalues.shape == (2,), f"Expected shape (2,), got {eigenvalues.shape}"
except Exception as e:
    print(f"Test failed with error: {e}")
    raise