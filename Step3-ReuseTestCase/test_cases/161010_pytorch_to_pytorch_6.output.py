import torch

# Ensure reproducibility
torch.manual_seed(0)

# Create a symmetric positive definite matrix for lobpcg
# lobpcg requires a symmetric positive definite matrix
A = torch.randn(5, 5, device="cuda" if torch.cuda.is_available() else "cpu")
A = A @ A.T + torch.eye(5, device=A.device)

def f(A, count):
    # Use torch.lobpcg instead of the original linear algebra chain
    eigenvalues, eigenvectors = torch.lobpcg(A, k=2)

    # Check stride preservation similar to the original bug report
    # We check the eigenvectors tensor
    if eigenvectors.stride() == eigenvectors.clone(memory_format=torch.preserve_format).stride():
        return count + 1
    return count

# Eager execution
res1 = f(A, torch.zeros(1))
print(f"Eager result: {res1}")

# Compiled execution
res2 = torch.compile(f)(A, torch.zeros(1))
print(f"Compiled result: {res2}")

# Verify that compiled behavior matches eager behavior
assert torch.equal(res1, res2), f"torch.compile mismatch: eager={res1}, compiled={res2}"