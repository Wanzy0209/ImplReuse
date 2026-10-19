import torch
import tensorflow as tf

# Replacing torch.nn.EmbeddingBag with tf.keras.backend.ndim
embedding_sum = tf.keras.backend.ndim

# Replacing torch.tensor with torch.isinf
# Replacing torch.long with tf.keras.backend.ndim (omitted as dtype is not supported by isinf)
input = torch.isinf([[1, 2, 4, 5], [4, 3, 2, 9]])

# Replacing the call embedding_sum(input)
# Converting torch tensor to tensorflow tensor for compatibility
result = embedding_sum(tf.convert_to_tensor(input))

# The bug is related to 2D inputs, so we verify the dimension is 2
assert result == 2