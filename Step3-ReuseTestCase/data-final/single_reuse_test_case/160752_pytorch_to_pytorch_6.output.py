import torch

BATCH = 37
MAX = 3

# Setup inputs adapted for torch.lobpcg
# We need a symmetric positive definite matrix A
x = torch.rand((BATCH, MAX), dtype=torch.float64)
# Create a symmetric matrix A (BATCH x BATCH)
A = x @ x.T + torch.eye(BATCH, dtype=torch.float64) * 0.1
# Number of eigenvalues to compute
k = 2
# Initial guess for eigenvectors (BATCH x k)
X = torch.rand((BATCH, k), dtype=torch.float64)

def lobpcg_func(A, X):
    # Call the similar API: torch.lobpcg
    return torch.lobpcg(A, k=k, X=X)

# Test uncompiled
eigs, vecs = lobpcg_func(A, X)

# Test compiled with dynamic=True (mimicking the original bug's trigger)
compiled_func = torch.compile(lobpcg_func, dynamic=True)
eigs_compiled, vecs_compiled = compiled_func(A, X)

# Verify results match
assert torch.allclose(eigs, eigs_compiled, atol=1e-4)
assert torch.allclose(vecs, vecs_compiled, atol=1e-4)