```python
import tensorflow as tf

# Conversion: torch.nn.EmbeddingBag with mode='sum' is equivalent to
# tf.keras.layers.Embedding followed by tf.reduce_sum.
# The 'include_last_offset' argument is specific to PyTorch's 1D input+offsets mode
# and is ignored when passing a 2D input (batch mode).
embedding_layer = tf.keras.layers.Embedding(input_dim=10, output_dim=3)

input_tensor = tf.constant([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=tf.int64)

# Look up embeddings: shape becomes (batch_size, sequence_length, embedding_dim)
embeddings = embedding_layer(input_tensor)

# Sum embeddings along the sequence dimension (axis=1) to match mode='sum'
output = tf.reduce_sum(embeddings, axis=1)
```