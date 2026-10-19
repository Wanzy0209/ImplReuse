```python
import tensorflow as tf

# PyTorch: torch.nn.EmbeddingBag(10, 3, mode='sum', include_last_offset=True)
# TensorFlow equivalent: Use Embedding layer followed by reduce_sum.
# Note: include_last_offset=True affects offset handling in PyTorch but does not change
# the output of the sum operation when using a 2D input tensor.
embedding_sum = tf.keras.layers.Embedding(input_dim=10, output_dim=3)

# PyTorch: input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long)
# TensorFlow uses int32 or int64 for indices.
input = tf.constant([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=tf.int64)

# PyTorch: embedding_sum(input)
# In TensorFlow, we perform the lookup and then sum the embeddings along the sequence axis.
# EmbeddingBag with mode='sum' on a 2D input is equivalent to summing the embeddings of each row.
tf.reduce_sum(embedding_sum(input), axis=1)
```