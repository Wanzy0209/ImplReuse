```python
# pyright: strict
import tensorflow as tf

# Conversion: torch.no_grad() disables gradient calculation.
# TensorFlow does not have a direct decorator equivalent.
# Gradients are tracked only within tf.GradientTape.
# To mimic the behavior of torch.no_grad (preventing gradient flow),
# we use tf.stop_gradient on the result.
def double(x: tf.Tensor) -> tf.Tensor:
    return tf.stop_gradient(2 * x)

reveal_type(double)
```