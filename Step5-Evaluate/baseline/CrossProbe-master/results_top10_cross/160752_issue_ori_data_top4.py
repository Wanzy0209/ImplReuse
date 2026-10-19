```python
import tensorflow as tf

MAX = 3
BATCH = 37


# Conversion: torch.nn.functional.one_hot -> tf.one_hot
# Conversion: x.square() -> tf.square(x)
def func(x, idxs):
    return tf.square(x) * tf.one_hot(idxs, MAX, dtype=x.dtype)


# Conversion: torch.func.jacfwd -> tf.jacobian
# Note: tf.jacobian computes the full Jacobian. Since it expects a function of a single variable,
# we wrap the call to pass 'idxs' as a non-differentiable argument.
def jacfunc(x, idxs):
    def f(x_input):
        return func(x_input, idxs)
    return tf.jacobian(f, x)


# Conversion: torch.randint -> tf.random.uniform (with int64 dtype)
idxs = tf.random.uniform((BATCH,), minval=0, maxval=MAX, dtype=tf.int64)
# Conversion: torch.rand -> tf.random.uniform
x = tf.random.uniform((BATCH, MAX), dtype=tf.float64)

# works
out = jacfunc(x, idxs)

# fails (in original PyTorch code, this was the compilation step)
# Conversion: torch.compile -> tf.function
jacfunc = tf.function(jacfunc)
out = jacfunc(x, idxs)
```