import tensorflow as tf

# Enable eager execution. This must be called at program startup and before
# any other TensorFlow operations. This corresponds to the eager execution
# context in the original PyTorch test case.
tf.compat.v1.enable_eager_execution()

def inner(x):
    return x + 1

def fn(x):
    x = inner(x)
    return inner(x)

# Execute the function with a tensor of ones
result = fn(tf.ones(3))

# Verify the result to ensure the eager execution worked as expected
# Expected result: [2, 2, 2]
assert tf.reduce_all(result == tf.constant([2, 2, 2])).numpy()