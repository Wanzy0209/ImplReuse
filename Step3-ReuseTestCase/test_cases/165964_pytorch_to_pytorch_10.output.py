import torch

# Create a small matrix on CUDA
# QR decomposition requires at least a 2D tensor, so we use (1, 1) to keep it minimal
input_tensor = torch.randn(1, 1, device="cuda")

# Perform QR decomposition
# Using torch.linalg.qr as it is the underlying implementation for torch.qr
Q, R = torch.linalg.qr(input_tensor)

# Verify the result is accessible (similar to .item() in the original test)
print(Q[0, 0].item())