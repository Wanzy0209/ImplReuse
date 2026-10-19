```python
import tensorflow as tf

# PyTorch EmbeddingBag with 2D input sums embeddings for each row.
# include_last_offset is ignored when input is 2D.
# We use Embedding layer followed by reduce_sum to mimic mode='sum'.
embedding_sum = tf.keras.layers.Embedding(input_dim=10, output_dim=3)

input = tf.constant([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=tf.int64)

# Apply embedding lookup and sum along the sequence dimension (axis=1)
# to replicate EmbeddingBag behavior with mode='sum'
tf.reduce_sum(embedding_sum(input), axis=1)
```