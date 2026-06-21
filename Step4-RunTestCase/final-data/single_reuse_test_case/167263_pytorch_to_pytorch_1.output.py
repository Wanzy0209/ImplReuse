import torch

# Setup dimensions
# B must be 1 for the view (B*N, N) to result in a square matrix (N, N)
# which is required for eigenvalue decomposition.
B, N = 1, 4

# Create a symmetric matrix input
# torch.linalg.eigh requires the input to be symmetric (or Hermitian)
x = torch.randn(B, N, N, requires_grad=True)
x = (x + x.transpose(-2, -1)) / 2

# Create a view to mimic the original scenario where stride issues might occur
# Original: (B, H, W, C) -> (B, H*W, C)
# Here: (B, N, N) -> (B*N, N)
# With B=1, this becomes (1, 4, 4) -> (4, 4), which is a valid square matrix.
x_view = x.view(B * N, N)

def log_grad(name):
    def hook(grad):
        print(f"{name} hook - shape: {grad.shape}, stride: {grad.stride()}")
        return grad
    return hook

print("forward strides", x_view.shape, x_view.stride())
x_view.register_hook(log_grad("x_view"))

# Call torch.linalg.eigh (replaces deprecated torch.symeig)
# Returns eigenvalues and eigenvectors. We use eigenvalues for the backward pass.
eigenvalues, _ = torch.linalg.eigh(x_view)

# Backward pass
eigenvalues.backward(torch.ones_like(eigenvalues))