```python
import tensorflow as tf

# Conversion: torch.nn.EmbeddingBag(10, 3, mode='sum', include_last_offset=True)
# TensorFlow does not have a direct EmbeddingBag layer. 
# We use tf.keras.layers.Embedding followed by tf.reduce_sum to simulate mode='sum'.
# include_last_offset=True is implicit when processing 2D input row-wise.
embedding_sum = tf.keras.layers.Embedding(input_dim=10, output_dim=3)

# Conversion: torch.tensor -> tf.constant
input = tf.constant([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=tf.int64)

# Conversion: embedding_sum(input)
# PyTorch EmbeddingBag performs lookup and reduction (sum) in one step.
# In TF, we lookup embeddings and then reduce along the sequence axis (axis=1).
tf.reduce_sum(embedding_sum(input), axis=1)
```