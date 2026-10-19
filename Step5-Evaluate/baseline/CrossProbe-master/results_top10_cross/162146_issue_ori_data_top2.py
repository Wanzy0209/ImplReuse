```python
import tensorflow as tf

# Conversion: Set random seed
tf.random.set_seed(2025)


def foo(x):
    # Conversion: TF tensors are immutable. We compute the sine of the rows
    # instead of modifying x in-place.
    # PyTorch: x[0].sin_()
    row0 = tf.math.sin(x[0])
    # PyTorch: x[1].sin_()
    row1 = tf.math.sin(x[1])

    # PyTorch: y = torch.zeros_like(x)
    y = tf.zeros_like(x)

    # PyTorch: y[2] = x[0]
    # PyTorch: y[3] = x[1]
    # Conversion: Use tensor_scatter_nd_update to mimic index assignment
    indices = tf.constant([[2], [3]])
    updates = tf.stack([row0, row1])
    y = tf.tensor_scatter_nd_update(y, indices, updates)

    return y


# Conversion: tf.function compiles the function into a static graph
cfoo = tf.function(foo)

# PyTorch: x = torch.tensor(...)
x = tf.constant([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=tf.float32)
cx = tf.constant([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=tf.float32)

res = foo(x)
cres = cfoo(cx)

# Conversion: Use tf.debugging.assert_near to check closeness
tf.debugging.assert_near(res, cres)
```