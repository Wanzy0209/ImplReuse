```python
import tensorflow as tf

MAX = 3
BATCH = 37


def func(x, idxs):
    # Conversion: torch.nn.functional.one_hot -> tf.one_hot
    # Conversion: x.square() -> tf.square(x)
    return tf.square(x) * tf.one_hot(idxs, MAX)


def jacfunc(x, idxs):
    # Conversion: torch.func.jacfwd -> tf.GradientTape.jacobian
    # Note: PyTorch's jacfwd uses forward-mode AD, while tf.GradientTape uses reverse-mode.
    # The resulting Jacobian tensor structure is equivalent.
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(x)
        y = func(x, idxs)
    return tape.jacobian(y, x)


# Conversion: torch.randint -> tf.random.uniform with dtype int64
idxs = tf.random.uniform((BATCH,), minval=0, maxval=MAX, dtype=tf.int64)

# Conversion: torch.rand -> tf.random.uniform with dtype float64
x = tf.random.uniform((BATCH, MAX), dtype=tf.float64)

# works
out = jacfunc(x, idxs)

# fails
# Conversion: torch.compile -> tf.function
jacfunc = tf.function(jacfunc)
out = jacfunc(x, idxs)
```