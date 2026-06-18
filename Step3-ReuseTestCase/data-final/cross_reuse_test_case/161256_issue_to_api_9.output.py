import torch
import torch.nn.functional as F

# Reproduce the environment settings from the bug report
torch.set_default_dtype(torch.float64)

# Create large float64 tensors on CUDA, similar to the original issue
x = torch.randn(1000, 1000, device="cuda")

# Apply the similar API (tanh) instead of matmul
z = F.tanh(x)

# Access the result to ensure the operation completes and data is accessible
# This mirrors the original print(z[0, 0])
print(z[0, 0])