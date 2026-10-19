```python
import tensorflow as tf


# Conversion: torch.full((2, ), x, dtype=torch.float64)
# TensorFlow equivalent is tf.fill. Note that tf.fill infers dtype from the input value x.
def func_nojit(x):
    return tf.fill((2, ), x)


# Conversion: torch.compile(func_nojit)
# TensorFlow equivalent is tf.function, which compiles the function into a graph.
func_jit = tf.function(func_nojit)

for func in [func_nojit, func_jit]:
    # Conversion: torch.tensor(5.0, dtype=torch.float64)
    # TensorFlow equivalent is tf.constant.
    x1 = tf.constant(5.0, dtype=tf.float64)
    x2 = tf.constant(10.0, dtype=tf.float64)
    print(func(x1))
    print(func(x2))
```