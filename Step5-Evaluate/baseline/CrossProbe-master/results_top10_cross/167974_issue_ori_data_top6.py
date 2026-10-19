```python
import tensorflow as tf

# Conversion: torch.nn.EmbeddingBag with mode='sum' is equivalent to
# tf.keras.layers.Embedding followed by tf.reduce_sum.
# The 'include_last_offset' parameter is specific to PyTorch's offset handling
# and is ignored when the input is a 2D tensor (as in this case).
embedding_sum = tf.keras.layers.Embedding(input_dim=10, output_dim=3)

input = tf.constant([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=tf.int64)

# PyTorch's EmbeddingBag automatically sums the embeddings for each row.
# In TensorFlow, we explicitly perform the lookup and then the reduction.
output = tf.reduce_sum(embedding_sum(input), axis=1)
```