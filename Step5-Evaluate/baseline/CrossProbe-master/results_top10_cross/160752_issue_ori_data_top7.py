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
    # Note: torch.func.jacfwd computes the Jacobian with respect to argnums 0 (x)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = func(x, idxs)
    
    return tape.jacobian(y, x)


# Conversion: torch.randint -> tf.random.uniform
idxs = tf.random.uniform((BATCH,), minval=0, maxval=MAX, dtype=tf.int64)
# Conversion: torch.rand -> tf.random.uniform
x = tf.random.uniform((BATCH, MAX), dtype=tf.float64)

# works
out = jacfunc(x, idxs)

# fails
# Conversion: torch.compile -> tf.function
jacfunc = tf.function(jacfunc)
out = jacfunc(x, idxs)
```