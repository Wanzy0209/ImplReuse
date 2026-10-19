```python
import tensorflow as tf

# PyTorch EmbeddingBag with mode='sum' is equivalent to
# tf.keras.layers.Embedding followed by tf.reduce_sum.
# The 'include_last_offset' parameter is specific to the offsets input method
# and is ignored when passing a 2D tensor input.
embedding_sum = tf.keras.layers.Embedding(input_dim=10, output_dim=3)

input_tensor = tf.constant([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=tf.int64)

# Perform embedding lookup and sum along the sequence dimension (axis=1)
# to replicate the 'sum' mode of EmbeddingBag.
embeddings = embedding_sum(input_tensor)
output = tf.reduce_sum(embeddings, axis=1)
```