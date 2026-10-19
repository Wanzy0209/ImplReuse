```python
import tensorflow as tf


# Conversion: torch.full((2, ), x, dtype=torch.float64)
# In TensorFlow, we can achieve this by creating a ones tensor of the desired dtype and multiplying by x.
# This ensures the output shape is (2,) and the dtype is enforced.
def func_nojit(x):
    return tf.ones((2,), dtype=tf.float64) * x


# Conversion: torch.compile(func_nojit)
# In TensorFlow, the equivalent mechanism for compiling a function into a graph is tf.function.
func_jit = tf.function(func_nojit)

for func in [func_nojit, func_jit]:
    # Conversion: torch.tensor(5.0, dtype=torch.float64)
    # In TensorFlow, we use tf.constant to create a tensor with a specific value and dtype.
    x1 = tf.constant(5.0, dtype=tf.float64)
    x2 = tf.constant(10.0, dtype=tf.float64)
    print(func(x1))
    print(func(x2))
```