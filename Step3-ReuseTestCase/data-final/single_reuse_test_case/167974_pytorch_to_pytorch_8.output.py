import torch

# Test case for torch.nn.GroupNorm
# Adapted from the EmbeddingBag issue to verify handling of 2D inputs.
# The original bug involved incorrect offset generation with 2D inputs.
# Here we verify that GroupNorm processes 2D inputs (N, C) correctly.

# Initialize GroupNorm
# num_channels must be divisible by num_groups
num_groups = 2
num_channels = 4
gn = torch.nn.GroupNorm(num_groups, num_channels)

# Create a 2D input tensor (Batch size 2, Channels 4)
# Note: GroupNorm expects float inputs, unlike EmbeddingBag which expects long indices
input = torch.tensor([[1.0, 2.0, 4.0, 5.0], [4.0, 3.0, 2.0, 9.0]], dtype=torch.float)

# Run the forward pass
output = gn(input)

# Verify the output shape matches the input shape
assert output.shape == input.shape, f"Shape mismatch: expected {input.shape}, got {output.shape}"

print("Test passed.")