```python
# pyright: strict
import tensorflow as tf

# Conversion Note: @torch.no_grad() disables gradient calculation in PyTorch.
# In TensorFlow 2.x, gradients are only tracked within a tf.GradientTape context.
# To explicitly prevent gradient flow through this operation (mimicking no_grad),
# we use tf.stop_gradient.
def double(x: tf.Tensor) -> tf.Tensor:
    return tf.stop_gradient(2 * x)

reveal_type(double)
```