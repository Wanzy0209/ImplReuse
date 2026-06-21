import torch
import torch.nn.functional as F

# Prepare inputs for torch.nn.functional.embedding
# input_indices: tensor containing indices to look up
input_indices = torch.randint(0, 10, (1, 3))
# weight: the embedding matrix (num_embeddings, embedding_dim)
weight = torch.randn(10, 5)

# The extreme value from the original bug report (int64 max - 4)
extreme_padding_idx = 9223372036854775803

# Call the similar API with the extreme value
# Original API: torch.nn.LazyConv1d(..., padding=extreme_value)
# Similar API: torch.nn.functional.embedding(..., padding_idx=extreme_value)
try:
    output = F.embedding(input_indices, weight, padding_idx=extreme_padding_idx)
    print("Test passed (no crash). Output shape:", output.shape)
except Exception as e:
    print(f"Caught exception: {type(e).__name__}: {e}")