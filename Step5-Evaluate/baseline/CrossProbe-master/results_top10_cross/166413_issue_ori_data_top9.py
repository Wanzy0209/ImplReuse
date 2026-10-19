```python
# pyright: strict
import tensorflow as tf

# Conversion: @torch.no_grad() disables gradient calculation.
# In TensorFlow, gradients are tracked only within tf.GradientTape.
# @tf.function is used here to compile the function into a graph,
# which is the standard optimization for inference in TF.
@tf.function
def double(x: tf.Tensor) -> tf.Tensor:
    return 2 * x

# reveal_type is a static analysis helper and not valid runtime Python.
# It is commented out to ensure the code is valid for runtime execution.
# reveal_type(double)
```