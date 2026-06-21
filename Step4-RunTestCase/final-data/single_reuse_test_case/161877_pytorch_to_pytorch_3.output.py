import torch
from torch.nn import EmbeddingBag

# Input data for EmbeddingBag
input_indices = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long)
offsets = torch.tensor([0, 4], dtype=torch.long)

# Initialize EmbeddingBag with a valid value.
# The original value (9223372036854775803) caused a storage size overflow.
# The maximum index in input_indices is 9, so num_embeddings must be at least 10.
model = EmbeddingBag(num_embeddings=10, embedding_dim=10)

# Execute the forward pass
output = model(input_indices, offsets)