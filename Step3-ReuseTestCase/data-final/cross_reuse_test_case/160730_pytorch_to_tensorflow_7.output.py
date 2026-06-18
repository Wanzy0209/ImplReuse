import torch
import tensorflow as tf
import numpy as np

def foo(x):
    # Using the requested API: tf.keras.backend.name_scope
    # This provides a context for the operations, similar to how torch.compile
    # groups operations, though name_scope is primarily for graph organization.
    with tf.keras.backend.name_scope("tan_sin_scope"):
        t = tf.math.tan(x)
        # Mimic torch.expand
        e = tf.broadcast_to(t, (31, 51, 1))
        mean_val = tf.reduce_mean(e)

        # Mimic the conditional logic based on scalar value
        # In TensorFlow graph mode (tf.function), tf.cond is used for control flow
        # dependent on tensor values.
        out1 = tf.cond(mean_val > 0.5,
                       lambda: tf.subtract(e, e * 0.5),
                       lambda: tf.add(e, e * 0.5))

        return tf.math.sin(out1)

# Setup data matching the original bug report
np.random.seed(0)
x = np.random.uniform(0, 10, size=(31, 51, 1)).astype(np.float16)
input_tensor = tf.constant(x)

# Run eager mode
eager_res = foo(input_tensor)

# Run compiled mode
# tf.function is the TensorFlow equivalent to torch.compile for optimization/JIT
compiled_foo = tf.function(foo)
compile_res = compiled_foo(input_tensor)

# Verify results match
# This assertion checks for the "silent computation error" described in the issue
np.testing.assert_allclose(eager_res.numpy(), compile_res.numpy(), rtol=1e-5, atol=1e-5)
print("Test passed: Eager and compiled results match.")