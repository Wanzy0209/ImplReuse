```python
import tensorflow as tf

MAX = 3
BATCH = 37


# Conversion: torch.nn.functional.one_hot -> tf.one_hot
# Conversion: x.square() -> tf.square(x)
def func(x, idxs):
    return tf.square(x) * tf.one_hot(idxs, MAX, dtype=x.dtype)


# Conversion: torch.func.jacfwd -> tf.jacobian
# Note: tf.jacobian computes the full Jacobian. We use a lambda to pass 'idxs' as a non-differentiable argument.
def jacfunc(x, idxs):
    return tf.jacobian(lambda x_in: func(x_in, idxs), x)


# Conversion: torch.randint -> tf.random.uniform (with int64 dtype)
idxs = tf.random.uniform((BATCH,), minval=0, maxval=MAX, dtype=tf.int64)

# Conversion: torch.rand -> tf.random.uniform
x = tf.random.uniform((BATCH, MAX), minval=0.0, maxval=1.0, dtype=tf.float64)

# works
out = jacfunc(x, idxs)

# fails
# Conversion: torch.compile -> tf.function
jacfunc = tf.function(jacfunc)
out = jacfunc(x, idxs)
```