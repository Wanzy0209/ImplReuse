```python
import tensorflow as tf

# PyTorch's EmbeddingBag with mode='sum' is equivalent to an Embedding layer
# followed by a sum reduction over the sequence length.
# Note: include_last_offset=True is only relevant when using 1D input with offsets.
# Since the input here is 2D, it implies bags per row.
embedding_sum = tf.keras.layers.Embedding(input_dim=10, output_dim=3)

input_tensor = tf.constant([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=tf.int64)

# Look up embeddings for all indices
embeddings = embedding_sum(input_tensor)

# Sum embeddings for each bag (row) to match mode='sum'
result = tf.reduce_sum(embeddings, axis=1)
```