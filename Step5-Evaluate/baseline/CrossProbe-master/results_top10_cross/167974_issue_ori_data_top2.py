```python
import tensorflow as tf

# Conversion: torch.nn.EmbeddingBag with mode='sum' is equivalent to
# an Embedding layer followed by a reduce_sum operation over the sequence axis.
# include_last_offset=True is ignored when input is 2D (treated as fixed-length bags).
embedding_sum = tf.keras.layers.Embedding(input_dim=10, output_dim=3)
input = tf.constant([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=tf.int64)

# Perform embedding lookup and sum embeddings for each row (bag)
tf.reduce_sum(embedding_sum(input), axis=1)
```