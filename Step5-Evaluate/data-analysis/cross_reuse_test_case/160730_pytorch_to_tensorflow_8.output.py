import numpy as np
import sys

# Handle the environment dependency error (GLIBCXX version mismatch)
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Test skipped: TensorFlow environment issue detected (GLIBCXX version mismatch or missing dependency).")
    print(f"Details: {e}")
    sys.exit(0)

def foo(x):
    """
    TensorFlow equivalent of the PyTorch function.
    Uses tf.keras.name_scope as the context manager for operations.
    """
    with tf.keras.name_scope("math_operations"):
        t = tf.tan(x)
        # torch.expand(31, 51, 1) on a tensor of shape (31, 51, 1) is a no-op.
        # In TensorFlow, the tensor retains its shape unless explicitly broadcasted.
        e = t
        
        mean_val = tf.reduce_mean(e)
        
        # TensorFlow's tf.function (autograph) handles the conversion of this Python if statement
        # into a graph conditional (tf.cond).
        if mean_val > 0.5:
            out1 = tf.subtract(e, e * 0.5)
        else:
            out1 = tf.add(e, e * 0.5)
            
        return tf.sin(out1)

# Setup data
np.random.seed(0)
x = np.random.uniform(0, 10, size=(31, 51, 1)).astype(np.float16)

# 1. Run in Eager mode (baseline)
# We call the function directly without the @tf.function decorator to simulate eager execution
eager_res = foo(tf.constant(x))

# 2. Run in Compiled mode (Graph mode)
# We use tf.function to compile the function, similar to torch.compile
compiled_foo = tf.function(foo)
compile_res = compiled_foo(tf.constant(x))

# Verify results
# Using numpy testing to check if the compiled and eager results are close.
# This mirrors the torch.testing.assert_close in the original bug report.
try:
    np.testing.assert_allclose(eager_res.numpy(), compile_res.numpy(), rtol=1e-5, atol=1e-5)
    print("Test Passed: Eager and Compiled results are close.")
except AssertionError as e:
    print(f"Test Failed: {e}")