```python
import tensorflow as tf

# PyTorch: torch.aminmax returns a named tuple (min, max).
# TensorFlow: Use tf.reduce_min and tf.reduce_max.
# Note: TensorFlow does not have a direct 'aminmax' function.
# We compute both values.
t = tf.constant([1, -3, 5])
min_val = tf.reduce_min(t)
max_val = tf.reduce_max(t)

# The original code included the return value representation.
# We can mimic this structure using a dictionary or simply print.
# Here we print the result to match the output visibility.
print(f"min={min_val}, max={max_val}")
```