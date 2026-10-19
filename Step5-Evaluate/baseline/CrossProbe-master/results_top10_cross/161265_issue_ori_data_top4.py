```python
import tensorflow as tf

# Conversion: torch.ones -> tf.ones
# Conversion: torch.int8 -> tf.int8
# Note: device='mps' is PyTorch specific; TF uses default device or tf.device context
a = tf.ones((2, (1 << 31) + 5), dtype=tf.int8)

# Conversion: Indexing works similarly in TF 2.x
print(a[1, -2])
print(a[:, -2])
```