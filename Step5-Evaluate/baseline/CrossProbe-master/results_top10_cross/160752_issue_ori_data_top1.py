```python
import tensorflow as tf

MAX = 3
BATCH = 37


# Conversion: torch.nn.functional.one_hot -> tf.one_hot
# Conversion: x.square() -> tf.square(x)
def func(x, idxs):
    return tf.square(x) * tf.one_hot(idxs, MAX, dtype=x.dtype)


# Conversion: torch.func.jacfwd -> tf.jacobian
# Note: torch.func.jacfwd returns a function, tf.jacobian computes the value directly.
def jacfunc(x, idxs):
    return tf.jacobian(func(x, idxs), x)


# Conversion: torch.randint -> tf.random.uniform
idxs = tf.random.uniform((BATCH,), minval=0, maxval=MAX, dtype=tf.int64)

# Conversion: torch.rand -> tf.random.uniform
x = tf.random.uniform((BATCH, MAX), dtype=tf.float64)

# works
out = jacfunc(x, idxs)

# fails (in PyTorch) / compiled (in TensorFlow)
# Conversion: torch.compile -> tf.function
jacfunc = tf.function(jacfunc)
out = jacfunc(x, idxs)
```