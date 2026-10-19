```python
# pyright: strict
import tensorflow as tf

# Conversion: @torch.no_grad() is not required in TensorFlow eager execution
# as gradients are not tracked by default. To explicitly stop gradients within
# a gradient tape context, use tf.stop_gradient.
def double(x: tf.Tensor) -> tf.Tensor:
    return 2 * x

# reveal_type is a static type checking utility (e.g., for pyright/mypy)
# and is not a runtime function. It is commented out for valid execution.
# reveal_type(double)
```