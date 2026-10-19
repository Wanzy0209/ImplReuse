```python
import tensorflow as tf

def f(x, max_val):
    # Conversion: torch.clamp(x, min, max) -> tf.clip_by_value(x, clip_value_min, clip_value_max)
    # Note: PyTorch's max_val.item() extracts a Python scalar. 
    # In TensorFlow, we can pass the scalar tensor directly to the operation.
    y = tf.clip_by_value(x, 0.0, max_val)
    return y

# Conversion: torch.compile -> tf.function
# jit_compile=True is used to enable XLA compilation, similar to the backend optimization in torch.compile.
compiled_func = tf.function(f, jit_compile=True)

# Conversion: torch.randn -> tf.random.normal
# Note: TensorFlow handles device placement (e.g., GPU) automatically.
x = tf.random.normal((10, 20, 30))
max_val = tf.constant(5.0)

compiled_func(x, max_val)
```