```python
# pyright: strict
import tensorflow as tf

# Conversion: @torch.no_grad() disables gradient calculation.
# In TensorFlow, gradients are tracked via tf.GradientTape.
# To prevent gradient flow through this operation, we use tf.stop_gradient.
def double(x: tf.Tensor) -> tf.Tensor:
    return tf.stop_gradient(2 * x)

# reveal_type is a static type checking directive (e.g., for pyright/mypy).
# It is not a runtime function, so it is commented out to ensure valid execution.
# reveal_type(double)
```