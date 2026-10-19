```python
# pyright: strict
import tensorflow as tf

# Conversion: @torch.no_grad() disables gradient calculation.
# In TensorFlow, the equivalent is wrapping the operation in tf.stop_gradient()
# to prevent gradient tracking through this specific computation.
def double(x: tf.Tensor) -> tf.Tensor:
    return tf.stop_gradient(2 * x)

reveal_type(double)
```