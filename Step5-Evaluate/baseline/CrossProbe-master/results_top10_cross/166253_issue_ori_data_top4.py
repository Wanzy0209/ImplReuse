```python
import tensorflow as tf

# Conversion: torch.full((2, ), x, dtype=torch.float64) -> tf.fill([2], x)
# Note: tf.fill creates a tensor with the given shape filled with the value.
def func_nojit(x):
    return tf.fill([2], x)

# Conversion: torch.compile(func_nojit) -> tf.function(func_nojit)
# tf.function compiles a Python function into a callable TensorFlow graph.
func_jit = tf.function(func_nojit)

for func in [func_nojit, func_jit]:
    # Conversion: torch.tensor(value, dtype) -> tf.constant(value, dtype)
    x1 = tf.constant(5.0, dtype=tf.float64)
    x2 = tf.constant(10.0, dtype=tf.float64)
    print(func(x1))
    print(func(x2))
```