```python
import tensorflow as tf

# Conversion: torch.nn.EmbeddingBag with mode='sum' and 2D input is equivalent to
# an Embedding layer followed by a reduce_sum operation over the sequence axis.
# Note: 'include_last_offset' is ignored in PyTorch when input is 2D.
embedding_layer = tf.keras.layers.Embedding(input_dim=10, output_dim=3)

# Conversion: torch.tensor with dtype=torch.long maps to tf.int64
# Renamed 'input' to 'input_tensor' to avoid shadowing Python built-in
input_tensor = tf.constant([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=tf.int64)

# Conversion: Perform embedding lookup and sum embeddings for each row
embeddings = embedding_layer(input_tensor)
output = tf.reduce_sum(embeddings, axis=1)
```