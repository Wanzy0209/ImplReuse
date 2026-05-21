import torch
import torch._dynamo

# Setup inputs for torch.cholesky_solve
# cholesky_solve(input, A) requires A to be a Cholesky factorization of a positive definite matrix.
# We create a random positive definite matrix and compute its Cholesky decomposition.
A = torch.randn(3, 3)
A = A @ A.T + torch.eye(3)  # Make symmetric positive definite
A_chol = torch.linalg.cholesky(A)

flag = True
dummy = lambda: None

def fn(x):
    x = x + 1
    torch._dynamo.graph_break()
    x = x + 2
    if flag:
        dummy.attr0 = x
    else:
        # Replace torch.no_grad with torch.cholesky_solve
        # to test the similar API in the same control flow context.
        res = torch.cholesky_solve(x, A_chol)
        dummy.attr1 = res
    return x + 4

# Input must be at least 2D for cholesky_solve
inp = torch.ones(3, 1)

opt_fn = torch.compile(fn, backend="eager")

# First run with flag=True
assert torch.allclose(fn(inp), opt_fn(inp))

# Second run with flag=False to trigger the else branch
flag = False
assert torch.allclose(fn(inp), opt_fn(inp))