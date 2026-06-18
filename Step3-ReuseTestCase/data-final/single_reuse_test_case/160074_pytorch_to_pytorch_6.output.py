import torch

# Define the function to be compiled, wrapping torch.lobpcg
def run_lobpcg(A):
    return torch.lobpcg(A)

# Compile with the same backend and settings as the original bug report
# (fullgraph=True, backend="inductor")
compiled_fn = torch.compile(run_lobpcg, fullgraph=True, backend="inductor")

with torch.device("cuda"):
    # Create a symmetric positive definite matrix A
    # Using bfloat16 and requires_grad=True to match the bug report's conditions
    # Shape is adapted to 2D for lobpcg (e.g., 128x128)
    X = torch.randn([128, 128], dtype=torch.bfloat16)
    A = X @ X.T + torch.eye(128, dtype=torch.bfloat16) * 1e-3
    A.requires_grad = True

    # Run the compiled function
    eigenvalues, eigenvectors = compiled_fn(A)

    # Run backward pass with random gradients, similar to the original reproducer
    eigenvalues.backward(torch.randn_like(eigenvalues))

    # Basic assertion to ensure gradients were computed
    assert A.grad is not None
    print("Test passed.")