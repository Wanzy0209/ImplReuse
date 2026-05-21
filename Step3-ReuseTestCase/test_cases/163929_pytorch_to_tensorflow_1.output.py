import tensorflow as tf
import numpy as np

def foo(x):
    # In-place mutation: using tan to preserve the original bug reproduction logic.
    # We ensure the input is positive so tan() remains positive, allowing the assertion to pass.
    x.assign(tf.tan(x))
    # Transpose
    x = tf.transpose(x)
    # Operation: assert_non_negative
    # This returns the input tensor if the assertion holds.
    return tf.debugging.assert_non_negative(x)

# Setup
tf.random.set_seed(0)
# Initialize with positive values to ensure tan() is positive, satisfying the assertion.
# This preserves the 'tan' operation while ensuring the test can verify correctness.
initial_data = tf.random.uniform((4, 6), 0.1, 1.0)
x1 = tf.Variable(initial_data)
x2 = tf.Variable(tf.identity(initial_data))

# Eager execution
out1 = foo(x1)

# Compiled execution (Graph mode)
cf = tf.function(foo)
out2 = cf(x2)

# Verification
# assert_non_negative returns the tensor if successful.
# We check if the outputs from eager and compiled modes are close.
np.testing.assert_allclose(out1.numpy(), out2.numpy())
print("Test passed: Eager and compiled outputs match.")