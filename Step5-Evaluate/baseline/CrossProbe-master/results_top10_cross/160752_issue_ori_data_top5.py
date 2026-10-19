```python
import tensorflow as tf

MAX = 3
BATCH = 37


def func(x, idxs):
    # Conversion: torch.nn.functional.one_hot -> tf.one_hot
    return tf.square(x) * tf.one_hot(idxs, MAX)


def jacfunc(x, idxs):
    # Conversion: torch.func.jacfwd -> tf.jacobian
    # We use a lambda to select the first argument (x) for differentiation
    return tf.jacobian(lambda x_val: func(x_val, idxs), x)


# Conversion: torch.randint -> tf.random.uniform + cast
idxs = tf.cast(tf.random.uniform((BATCH,), minval=0, maxval=MAX, dtype=tf.int32), tf.int64)
# Conversion: torch.rand -> tf.random.uniform
x = tf.random.uniform((BATCH, MAX), dtype=tf.float64)

# works
out = jacfunc(x, idxs)

# fails
# Conversion: torch.compile -> tf.function
jacfunc = tf.function(jacfunc)
out = jacfunc(x, idxs)
```