```python
# pyright: strict
import tensorflow as tf

# Conversion Note: @torch.no_grad() is not directly translatable to TensorFlow 2.x
# eager execution as gradients are not tracked by default. In TF, gradients are
# tracked explicitly via tf.GradientTape. If this function is used inside a
# context where gradients are tracked and you wish to stop them, use
# tf.stop_gradient(x).
def double(x: tf.Tensor) -> tf.Tensor:
    return 2 * x

# reveal_type is a static type checking command (pyright/mypy).
# It is commented out to ensure the code is valid at runtime.
# reveal_type(double)
```