import torch
import tensorflow as tf
import numpy as np

# Set seeds for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

def foo(input):
    # Adapted logic: using tf.nn.avg_pool3d instead of interpolate
    # Note: interpolate upsamples, avg_pool3d downsamples.
    # We are testing the consistency between eager and compiled execution.
    pool = tf.nn.avg_pool3d(
        input,
        ksize=2,
        strides=2,
        padding='SAME',
        data_format='NDHWC'
    )
    # Preserve post-processing logic: squeeze batch dim, then argmin
    squeeze = tf.squeeze(pool, axis=0)
    argmin = tf.argmin(squeeze, axis=1)
    return argmin

# Create input data
# Original input was (1, 40, 1, 1). avg_pool3d requires 5D input.
# We create a 5D tensor with similar magnitude.
x = np.random.uniform(0, 10, size=(1, 40, 40, 40, 1)).astype(np.float64)

# Eager execution
eager_res = foo(tf.convert_to_tensor(x))

# Compiled execution (simulating torch.compile with tf.function + XLA)
cfoo = tf.function(foo, jit_compile=True)
compile_res = cfoo(tf.convert_to_tensor(x))

# Verify consistency
np.testing.assert_allclose(eager_res.numpy(), compile_res.numpy())
print("Test passed.")