import torch
import torch.nn as nn

# Adapted test case for torch.nn.Conv2d based on the EmbeddingBag issue.
# Original API: torch.nn.EmbeddingBag(10, 3, mode='sum', include_last_offset=True)
# Similar API: torch.nn.Conv2d(in_channels, out_channels, kernel_size)
# We map the dimensions: embedding_dim(3) -> in_channels, num_embeddings(10) -> out_channels

conv = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)

# Original input: torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long) with shape (2, 4)
# Conv2d expects input of shape (Batch, Channels, Height, Width).
# We adapt the input to be a valid 4D tensor for Conv2d.
# Batch size = 2 (matches original), Channels = 3 (matches in_channels), Height = 4, Width = 4.
input_tensor = torch.randn(2, 3, 4, 4)

# Perform the forward pass
output = conv(input_tensor)

# Verify the output shape
# Formula: H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
# With defaults (padding=0, stride=1, dilation=1, kernel_size=3):
# H_out = (4 - 3) + 1 = 2
# W_out = (4 - 3) + 1 = 2
# Expected shape: (2, 10, 2, 2)
assert output.shape == (2, 10, 2, 2), f"Expected shape (2, 10, 2, 2), got {output.shape}"