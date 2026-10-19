import torch
import tensorflow as tf
import tf.experimental.numpy as tnp

# Replacement: nn.EmbeddingBag -> torch.export.Constraint
# Original: torch.nn.EmbeddingBag(10, 3, mode='sum', include_last_offset=True)
# Adapted to Constraint signature: torch.export.Constraint(dim, min, max)
# We map the original arguments (10, 3) to the new API's parameters.
embedding_sum = torch.export.Constraint(dim=10, min=3)

# Replacement: torch.long -> torch.export.Constraint
# We replace the dtype symbol with the Constraint class.
dtype = torch.export.Constraint

# Replacement: torch.tensor -> tf.experimental.numpy.diag
# Original: torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long)
# To preserve the logic of handling 2D input (central to the bug), 
# we use diag on a 1D array to generate a 2D tensor.
# We use the flattened values from the original input.
input = tnp.diag([1, 2, 4, 5, 4, 3, 2, 9])

# Logic: The bug report concerns incorrect offsets when input is 2D.
# We leverage the similar API (tf.experimental.numpy.ndim) to verify 
# that the input is indeed 2D, preserving the context of the original issue.
assert tnp.ndim(input) == 2