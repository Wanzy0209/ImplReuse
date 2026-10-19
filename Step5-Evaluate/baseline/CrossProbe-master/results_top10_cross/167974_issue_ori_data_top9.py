```python
import tensorflow as tf

# Conversion: torch.nn.EmbeddingBag with mode='sum' is equivalent to
# tf.keras.layers.Embedding followed by tf.reduce_sum.
# Note: include_last_offset is ignored when input is a 2D tensor in PyTorch.
embedding_sum = tf.keras.layers.Embedding(input_dim=10, output_dim=3)

input = tf.constant([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=tf.int64)

# Conversion: PyTorch EmbeddingBag sums embeddings per row when given a 2D input.
# We perform the lookup and then reduce_sum along the sequence axis.
output = tf.reduce_sum(embedding_sum(input), axis=1)
```