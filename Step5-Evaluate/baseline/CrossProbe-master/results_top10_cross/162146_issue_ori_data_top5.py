```python
import tensorflow as tf
import numpy as np

# PyTorch: torch.manual_seed(2025)
# TensorFlow: tf.random.set_seed sets the global random seed
tf.random.set_seed(2025)


def foo(x):
    # PyTorch: x[0].sin_() (in-place operation)
    # TensorFlow: Tensors are immutable, so we compute the sine directly
    # without modifying the original tensor x.
    row0_sin = tf.math.sin(x[0])
    row1_sin = tf.math.sin(x[1])

    # PyTorch: y = torch.zeros_like(x)
    y = tf.zeros_like(x)

    # PyTorch: y[2] = x[0]; y[3] = x[1]
    # TensorFlow: Use tensor_scatter_nd_update to assign values to specific indices
    y = tf.tensor_scatter_nd_update(y, [[2]], [row0_sin])
    y = tf.tensor_scatter_nd_update(y, [[3]], [row1_sin])
    
    return y


# PyTorch: cfoo = torch.compile(foo)
# TensorFlow: tf.function compiles the function into a static graph for performance
cfoo = tf.function(foo)

# PyTorch: x = torch.tensor(...)
# TensorFlow: tf.constant creates a constant tensor
x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=tf.float32)
cx = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=tf.float32)

res = foo(x)
cres = cfoo(cx)

# PyTorch: torch.testing.assert_close(res, cres)
# TensorFlow: Use numpy's assert_allclose to verify numerical equality
np.testing.assert_allclose(res, cres)
```