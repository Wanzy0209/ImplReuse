import torch
import tensorflow as tf

def foo(x):
    """
    Adapted function to mimic the original bug scenario:
    1. In-place mutation (x.tan_())
    2. Transpose/View operation (x = x.t())
    3. Operation on the result (Target API: tf.debugging.assert_positive)
    """
    # Mimic in-place mutation x.tan_()
    # We use assign on a tf.Variable to simulate in-place modification
    x.assign(tf.tan(x))
    
    # Mimic transpose x = x.t()
    x = tf.transpose(x)
    
    # Target API: tf.debugging.assert_positive
    # This checks if all elements are positive.
    # If the compiler mishandles the mutation/transpose dependency,
    # it might check the wrong data (e.g., original data before tan).
    tf.debugging.assert_positive(x)
    
    return x

# Setup
tf.random.set_seed(0)
# Create a variable to allow in-place operations, mimicking the mutable tensor in PyTorch
initial_data = tf.random.normal((4, 6))
x1 = tf.Variable(initial_data)
x2 = tf.Variable(initial_data)

# Eager execution
eager_error = None
out1 = None
try:
    out1 = foo(x1)
except Exception as e:
    eager_error = e

# Compiled execution (tf.function)
# This is analogous to torch.compile in the original bug report
cf = tf.function(foo)
compiled_error = None
out2 = None
try:
    out2 = cf(x2)
except Exception as e:
    compiled_error = e

# Verification
# Check if the behavior (passing or raising error) is consistent between modes
if (eager_error is None) != (compiled_error is None):
    raise AssertionError(
        f"Behavior mismatch between eager and compiled execution.\n"
        f"Eager raised: {type(eager_error).__name__ if eager_error else None}\n"
        f"Compiled raised: {type(compiled_error).__name__ if compiled_error else None}"
    )

# If both succeeded, verify the outputs are identical
if eager_error is None:
    tf.debugging.assert_equal(out1, out2)
    print("Test passed: Eager and compiled execution produced consistent results.")
else:
    print(f"Test passed: Both eager and compiled execution raised {type(eager_error).__name__} consistently.")