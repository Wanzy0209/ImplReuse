```python
# pyright: strict
import tensorflow as tf

# Conversion: torch.no_grad() disables gradient calculation.
# In TensorFlow, gradients are tracked explicitly via GradientTape.
# To mimic the behavior of preventing gradients, we wrap the operation
# in tf.stop_gradient. There is no direct decorator equivalent in TF.
def double(x: tf.Tensor) -> tf.Tensor:
    return tf.stop_gradient(2 * x)

reveal_type(double)
```