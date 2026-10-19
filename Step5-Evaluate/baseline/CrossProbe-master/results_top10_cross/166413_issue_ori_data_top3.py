```python
# pyright: strict
import tensorflow as tf

# Conversion: @torch.no_grad() is not required in TensorFlow 2.x eager execution.
# Gradients are only tracked within a tf.GradientTape context.
# To explicitly stop gradients, wrap the result in tf.stop_gradient().
def double(x: tf.Tensor) -> tf.Tensor:
    return 2 * x

reveal_type(double)
```