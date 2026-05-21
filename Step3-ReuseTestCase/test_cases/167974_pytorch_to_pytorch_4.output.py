import torch
import torch.nn as nn

# Adapted test case for torch.nn.Conv3d based on the EmbeddingBag issue.
# The original issue involved a 2D input and a specific flag (include_last_offset).
# Since Conv3d requires 5D input and does not have an include_last_offset flag,
# we adapt the input dimensions to fit Conv3d and verify basic forward pass.

# Original input: torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long) -> Shape (2, 4)
# Adapted input for Conv3d: Shape (Batch, Channel, Depth, Height, Width) -> (2, 1, 1, 1, 4)
# We cast to float because Conv3d expects float inputs, unlike EmbeddingBag which expects indices (long).

conv3d = nn.Conv3d(in_channels=1, out_channels=3, kernel_size=1)
input_tensor = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.float).view(2, 1, 1, 1, 4)

output = conv3d(input_tensor)

# Verify the output shape is correct
# Input: (2, 1, 1, 1, 4), Kernel: 1, Stride: 1, Padding: 0
# Output: (2, 3, 1, 1, 4)
assert output.shape == (2, 3, 1, 1, 4), f"Expected shape (2, 3, 1, 1, 4), got {output.shape}"
print("Test passed.")