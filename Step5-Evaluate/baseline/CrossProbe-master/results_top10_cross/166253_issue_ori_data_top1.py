```python
import tensorflow as tf

# Conversion: torch.full((2, ), x, dtype=torch.float64) -> tf.fill([2], x, dtype=tf.float64)
def func_nojit(x):
    return tf.fill([2], x, dtype=tf.float64)

# Conversion: torch.compile(func_nojit) -> tf.function(func_nojit)
func_jit = tf.function(func_nojit)

for func in [func_nojit, func_jit]:
    # Conversion: torch.tensor(5.0, dtype=torch.float64) -> tf.constant(5.0, dtype=tf.float64)
    x1 = tf.constant(5.0, dtype=tf.float64)
    x2 = tf.constant(10.0, dtype=tf.float64)
    print(func(x1))
    print(func(x2))
```