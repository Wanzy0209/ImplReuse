import torch

# Adapted test case for torch.norm based on the torch.ones bug report.
# The original issue involved a CUDA out of memory error when creating a tensor on the GPU.
# This test verifies if torch.norm behaves correctly when operating on a CUDA tensor.

# Create a tensor on CUDA (mirroring the original failing call)
x = torch.ones(1, device="cuda")

# Apply the similar API: torch.norm
result = torch.norm(x)

# Verify the result is accessible
print(result.item())