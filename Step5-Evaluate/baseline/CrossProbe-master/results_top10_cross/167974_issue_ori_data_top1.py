```python
import tensorflow as tf

# PyTorch's EmbeddingBag with mode='sum' aggregates embeddings.
# In TensorFlow, this is implemented using an Embedding layer
# followed by a reduction operation (e.g., tf.reduce_sum).
# Note: include_last_offset is ignored for 2D input in PyTorch,
# treating each row as a separate bag.
embedding_sum = tf.keras.layers.Embedding(input_dim=10, output_dim=3)

# Input tensor
input = tf.constant([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=tf.int64)

# Apply embedding and sum over the sequence dimension (axis=1)
# to replicate the behavior of mode='sum'.
output = tf.reduce_sum(embedding_sum(input), axis=1)
```