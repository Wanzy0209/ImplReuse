```python
# pyright: strict
import tensorflow as tf

# Conversion: torch.no_grad() is not required in TensorFlow 2.x eager execution
# as gradients are only calculated within tf.GradientTape contexts.
# The provided target API 'analyze_test_numerics' is a test utility function
# and not a decorator, so it is not applicable here.
def double(x: tf.Tensor) -> tf.Tensor:
    return 2 * x

# reveal_type is used for static type checking (e.g., pyright, mypy)
reveal_type(double)
```