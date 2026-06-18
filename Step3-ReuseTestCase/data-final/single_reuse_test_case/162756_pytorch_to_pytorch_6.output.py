import torch

# Reproduce the environment settings from the bug report
torch._inductor.config.combo_kernels = True

# Prepare inputs for torch.lobpcg
# lobpcg requires a symmetric positive definite matrix
n = 16
A = torch.randn(n, n, device="cuda")
A = A @ A.T + torch.eye(n, device="cuda") * 0.1

@torch.compile
def fn(A):
    # Adapt the call site to use torch.lobpcg
    return torch.lobpcg(A, k=2)

# Run the test
try:
    eigvals, eigvecs = fn(A)
    # Verify output shapes
    assert eigvals.shape == (2,), f"Expected eigenvalues shape (2,), got {eigvals.shape}"
    assert eigvecs.shape == (n, 2), f"Expected eigenvectors shape ({n}, 2), got {eigvecs.shape}"
    print("Test passed.")
except Exception as e:
    print(f"Test failed: {e}")