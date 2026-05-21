import torch
import tensorflow as tf

def foo(x):
    # Mimic in-place mutation x.tan_()
    x.assign(tf.tan(x))
    # Mimic x = x.t()
    x = tf.transpose(x)
    # Target API: tf.compat.v1.assert_non_positive
    return tf.compat.v1.assert_non_positive(x)

# Setup
tf.random.set_seed(0)
# Use Variable to mimic mutable tensor behavior
x1 = tf.Variable(tf.random.normal((4, 6)))
x2 = tf.Variable(tf.random.normal((4, 6)))

# Test 1: Eager execution
try:
    out1 = foo(x1)
    eager_error = None
except tf.errors.InvalidArgumentError as e:
    eager_error = e

# Test 2: Compiled execution (tf.function)
# This mimics torch.compile(foo)
cf = tf.function(foo)
try:
    out2 = cf(x2)
    compiled_error = None
except tf.errors.InvalidArgumentError as e:
    compiled_error = e

# Verification
# Since tan(random_normal) produces positive values, assert_non_positive should fail in both modes.
# We verify that both eager and compiled modes raise the error consistently.
assert eager_error is not None, "Eager execution should raise InvalidArgumentError"
assert compiled_error is not None, "Compiled execution should raise InvalidArgumentError"

print("Test passed: Both eager and compiled modes correctly identified non-positive condition violation.")