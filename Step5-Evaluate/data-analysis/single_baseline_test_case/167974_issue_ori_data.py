# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
embedding_sum = torch.nn.EmbeddingBag(10, 3, mode='sum', include_last_offset=True)
input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long)
embedding_sum(input)