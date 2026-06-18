import torch

torch.manual_seed(2025)

# Create a symmetric positive definite matrix required for lobpcg
A = torch.randn(10, 10, dtype=torch.float32)
A = A @ A.T + torch.eye(10) * 0.1

def foo(A):
    # Call the similar API: torch.lobpcg
    # We request the largest 2 eigenvalues and corresponding eigenvectors
    eigenvalues, eigenvectors = torch.lobpcg(A, k=2)
    return eigenvalues, eigenvectors

# Compile the function containing the similar API
cfoo = torch.compile(foo)

# Execute both eager and compiled versions
res_e, res_v = foo(A)
cres_e, cres_v = cfoo(A)

# Verify that the results match
torch.testing.assert_close(res_e, cres_e)
torch.testing.assert_close(res_v, cres_v)