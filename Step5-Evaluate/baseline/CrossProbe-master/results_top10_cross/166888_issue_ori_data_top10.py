```python
import tensorflow as tf

def f(x, max_val):
    # Conversion: torch.clamp(x, 0, max_val.item()) -> tf.clip_by_value
    # Note: In TensorFlow, tf.clip_by_value accepts tensors directly for the bounds,
    # so we do not need to extract a scalar value like .item() in PyTorch.
    y = tf.clip_by_value(x, 0.0, max_val)
    return y

# Conversion: torch.compile -> tf.function
# tf.function compiles the function into a TensorFlow graph, analogous to torch.compile
compiled_func = tf.function(f)

# Conversion: torch.randn -> tf.random.normal
# Conversion: device='cuda' -> tf.device('/GPU:0')
with tf.device('/GPU:0'):
    x = tf.random.normal((10, 20, 30))
    # Conversion: torch.tensor -> tf.constant
    max_val = tf.constant(5.0)

compiled_func(x, max_val)
```