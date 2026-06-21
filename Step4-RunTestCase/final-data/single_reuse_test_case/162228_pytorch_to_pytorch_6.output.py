import torch

# Handle environments where torch.compile is not available (PyTorch < 2.0)
# We define a dummy decorator if torch.compile is missing to preserve the test logic.
if hasattr(torch, 'compile'):
    compile_decorator = torch.compile()
else:
    def compile_decorator(func):
        return func

@compile_decorator
def test(A, B):
    # lobpcg computes eigenvalues and eigenvectors.
    # We return the sum of eigenvalues to create a scalar for backward().
    eigenvalues, _ = torch.lobpcg(A, B=B, k=2)
    return eigenvalues.sum()

DEVICE = "cuda"
N = 10

# Create A and B as leaf tensors with requires_grad=True
A = torch.randn(N, N, device=DEVICE, dtype=torch.float64, requires_grad=True)
B = torch.randn(N, N, device=DEVICE, dtype=torch.float64, requires_grad=True)

# Make them Symmetric Positive Definite in-place to preserve leaf status
# This is necessary for lobpcg to converge properly.
A.data.copy_((A + A.T) / 2 + 10 * torch.eye(N, device=DEVICE, dtype=torch.float64))
B.data.copy_((B + B.T) / 2 + 10 * torch.eye(N, device=DEVICE, dtype=torch.float64))

# Run forward and backward
out = test(A, B)
out.backward()

print(torch.__version__)
print(f"A: {(A.grad is not None) and (A.grad.norm() > 0)}, B: {(B.grad is not None) and (B.grad.norm() > 0)}")

# Verify that gradients are populated for both inputs
assert A.grad is not None
assert A.grad.norm() > 0
assert B.grad is not None
assert B.grad.norm() > 0