import torch
from torch import lobpcg

# Configuration from the bug report
# Handle cases where torch._dynamo or torch._inductor might not be exposed or configured differently
try:
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch._inductor.config.emulate_precision_casts = True
except AttributeError:
    # If these specific internal configs aren't available, we skip them.
    # This allows the test to run on environments where torch._dynamo isn't directly exposed
    # or the specific config keys don't exist.
    pass

def test_lobpcg(A, B, X):
    # Replacing flex_attention with lobpcg
    # lobpcg returns (eigenvalues, eigenvectors)
    e, v = lobpcg(A, B=B, X=X)
    return e, v

# Setup inputs
# Adapted shapes: (27, 26, 122, 122) for matrices, (27, 26, 122, 10) for initial guess
# to match the batch dimensions and feature size of the original bug report.
batch_dims = (27, 26)
n = 122
k = 10
device = 'cuda'

# Create symmetric positive definite matrices A and B
A = torch.randn(*batch_dims, n, n, dtype=torch.float32, device=device)
A = (A + A.transpose(-2, -1)) / 2 + torch.eye(n, device=device) * n

B = torch.randn(*batch_dims, n, n, dtype=torch.float32, device=device)
B = (B + B.transpose(-2, -1)) / 2 + torch.eye(n, device=device) * n

X = torch.randn(*batch_dims, n, k, dtype=torch.float32, device=device)

# Run Eager
e_eager, v_eager = test_lobpcg(A, B, X)

# Run Compiled
compiled_test_lobpcg = torch.compile(test_lobpcg)
e_compiled, v_compiled = compiled_test_lobpcg(A, B, X)

# Verify
assert torch.allclose(e_eager, e_compiled, atol=1e-3)
assert torch.allclose(v_eager, v_compiled, atol=1e-3)