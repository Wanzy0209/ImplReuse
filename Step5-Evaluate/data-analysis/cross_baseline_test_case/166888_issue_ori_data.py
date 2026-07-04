```python
import tensorflow as tf

def f(x, max_val):
    # Conversion: torch.clamp(x, 0, max_val.item()) -> tf.clip_by_value(x, 0.0, max_val)
    # Note: max_val.item() extracts a scalar. In TF, clip_by_value handles tensor inputs for bounds.
    y = tf.clip_by_value(x, 0.0, max_val)
    return y

# Conversion: torch.compile -> tf.function
# Note: backend='inductor' and fullgraph=True are PyTorch specific optimizations.
# tf.function compiles the function into a static graph.
compiled_func = tf.function(f)

# Conversion: torch.randn -> tf.random.normal
# Conversion: device='cuda' -> Implicit in TF if GPU is available, or use tf.device context
x = tf.random.normal((10, 20, 30))
max_val = tf.constant(5.0)

compiled_func(x, max_val)
```