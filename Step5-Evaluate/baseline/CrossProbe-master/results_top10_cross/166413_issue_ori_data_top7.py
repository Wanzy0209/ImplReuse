```python
# pyright: strict
import tensorflow as tf

# Conversion: @torch.no_grad() disables gradient calculation within the decorated function.
# In TensorFlow, the tf.stop_gradient context manager is used to prevent
# gradient tracking within a specific scope.
def double(x: tf.Tensor) -> tf.Tensor:
    with tf.stop_gradient():
        return 2 * x

# reveal_type is a static type checking command (e.g., for pyright/mypy)
# It is preserved here for type checking purposes.
reveal_type(double)
```