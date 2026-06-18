import torch

def fn(A, k, X):
    # Using torch.lobpcg as the target API
    # This function finds the k largest eigenvalues and corresponding eigenvectors
    return torch.lobpcg(A, k=k, X=X)

# Setup inputs for lobpcg
# A must be a symmetric positive definite matrix
torch.manual_seed(42)
A = torch.randn(5, 5)
A = A @ A.T + torch.eye(5) # Ensure positive definiteness
k = 2
X = torch.randn(5, k)

# Apply torch.compile to the function using the similar API
# This mirrors the structure of the original test case
compiled_fn = torch.compile(fn, backend="eager")

# Execute the compiled function
eigenvalues, eigenvectors = compiled_fn(A, k, X)

# Verify the output shapes
assert eigenvalues.shape == (k,)
assert eigenvectors.shape == (A.shape[0], k)

print("Test passed.")