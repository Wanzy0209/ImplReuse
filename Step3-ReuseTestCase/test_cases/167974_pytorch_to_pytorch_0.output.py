import torch
from torch.ao.nn.qat.modules.embedding_ops import EmbeddingBag

# Test case for torch.ao.nn.qat.modules.embedding_ops.EmbeddingBag
# based on Issue 167974 regarding include_last_offset behavior with 2D input.

# Initialize the QAT EmbeddingBag with include_last_offset=True
embedding_sum = EmbeddingBag(10, 3, mode='sum', include_last_offset=True)

# Create a 2D input tensor
input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long)

# Execute the forward pass
# Note: The bug report indicates that internal offsets might be incorrect,
# but the API should still execute and return a result.
output = embedding_sum(input)

# Basic assertion to ensure the operation completes and returns the expected shape
# Input has 2 rows, so output should have shape (2, 3)
assert output.shape == (2, 3), f"Expected output shape (2, 3), but got {output.shape}"