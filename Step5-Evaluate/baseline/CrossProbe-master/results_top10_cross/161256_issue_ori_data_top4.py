```python
import tensorflow as tf

# Conversion: TensorFlow does not have a global set_default_dtype. 
# We explicitly pass dtype=tf.float64 to the tensor creation functions.
# torch.set_default_dtype(torch.float64)

# Conversion: Explicitly place tensors on GPU to match device="cuda"
with tf.device('/GPU:0'):
    # Conversion: torch.randn -> tf.random.normal
    x = tf.random.normal((1000, 1000), dtype=tf.float64)
    y = tf.random.normal((1000, 1000), dtype=tf.float64)

z = x @ y
print(z[0, 0])
```