import torch
import torch.nn as nn

# Test case for torch.nn.MaxUnpool3d with 0-shape dimension
# Based on Issue 168071: torch.nn.functional.pad crashes when padding (0, 0) to 0-shape
# This test verifies if MaxUnpool3d handles 0-shape dimensions correctly without crashing
# or reporting invalid negative sizes.

# Input tensor with a 0 dimension (Depth=0)
# Shape: (Batch, Channel, Depth, Height, Width)
x = torch.randn(1, 1, 0, 2, 2)
indices = torch.zeros(1, 1, 0, 2, 2, dtype=torch.long)

# Initialize MaxUnpool3d
# kernel_size=2, stride=2, padding=0
# Expected output size calculation:
# Depth: (0 - 1) * 2 - 2*0 + 2 = 0
# Height: (2 - 1) * 2 - 2*0 + 2 = 4
# Width: (2 - 1) * 2 - 2*0 + 2 = 4
# Expected Output Shape: (1, 1, 0, 4, 4)
unpool = nn.MaxUnpool3d(kernel_size=2, stride=2)

try:
    output = unpool(x, indices)
    print(f"Test Passed. Output shape: {output.shape}")
    assert output.shape == torch.Size([1, 1, 0, 4, 4])
except RuntimeError as e:
    print(f"Test Failed with RuntimeError: {e}")