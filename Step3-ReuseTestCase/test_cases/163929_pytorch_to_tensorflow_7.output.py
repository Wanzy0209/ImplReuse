import torch
import tensorflow as tf

def foo(x):
    # PyTorch: x.tan_() (in-place mutation)
    # TensorFlow: Tensors are immutable, so we apply the operation and reassign.
    x = tf.tan(x)

    # PyTorch: x = x.t() (transpose)
    # TensorFlow: Transpose the tensor.
    x = tf.transpose(x)

    # PyTorch: return x.argmin()
    # TensorFlow: tf.debugging.assert_negative checks if all values are < 0.
    # This operation raises an exception if the condition is not met.
    return tf.debugging.assert_negative(x)

# Setup
tf.random.set_seed(0)
# Create random data similar to torch.randn(4, 6)
x1 = tf.random.normal((4, 6))
x2 = tf.identity(x1) # Clone

# Run eager
eager_error = None
try:
    out1 = foo(x1)
except Exception as e:
    eager_error = type(e)

# Run compiled (tf.function equivalent to torch.compile)
cf = tf.function(foo)
compiled_error = None
try:
    out2 = cf(x2)
except Exception as e:
    compiled_error = type(e)

# Verify behavior consistency
# The original bug was a mismatch in output. Here we check for a mismatch in behavior (pass/fail).
if eager_error != compiled_error:
    raise AssertionError(
        f"Behavior differs between eager and compiled modes.\n"
        f"Eager error: {eager_error}\n"
        f"Compiled error: {compiled_error}"
    )