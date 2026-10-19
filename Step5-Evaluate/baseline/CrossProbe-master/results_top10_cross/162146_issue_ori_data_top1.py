```python
import tensorflow as tf
import numpy as np

# torch.manual_seed(2025) -> tf.random.set_seed(2025)
tf.random.set_seed(2025)


def foo(x):
    # x[0].sin_() -> In-place sine. TF tensors are immutable, so we compute the value.
    # x[1].sin_()
    r0 = tf.math.sin(x[0])
    r1 = tf.math.sin(x[1])

    # y = torch.zeros_like(x)
    y = tf.zeros_like(x)

    # y[2] = x[0] (which is now sin(x[0]))
    # y[3] = x[1] (which is now sin(x[1]))
    # Use tensor_scatter_nd_update to mimic assignment in a graph-compatible way
    indices = [[2], [3]]
    updates = tf.stack([r0, r1])
    y = tf.tensor_scatter_nd_update(y, indices, updates)

    return y


# cfoo = torch.compile(foo) -> cfoo = tf.function(foo)
cfoo = tf.function(foo)

# x = torch.tensor(...) -> tf.constant(...)
x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=tf.float32)
cx = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=tf.float32)

res = foo(x)
cres = cfoo(cx)

# torch.testing.assert_close(res, cres) -> np.testing.assert_allclose
np.testing.assert_allclose(res.numpy(), cres.numpy())
```