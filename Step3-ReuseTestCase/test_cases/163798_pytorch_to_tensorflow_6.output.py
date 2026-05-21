import torch
import tensorflow as tf

# The original test case uses torch.compile to graph the function.
# The similar API identified is tf.compat.v1.name_scope.
# We adapt the test by wrapping the logic in name_scope and executing it in graph mode (tf.function).

@tf.function
def func(a):
    # Using the similar API: tf.compat.v1.name_scope
    # This context manager handles naming within the graph, analogous to how torch.compile handles graph construction.
    with tf.compat.v1.name_scope("extraction_scope"):
        # Original logic: u0, u1 = a.tolist()
        # In TensorFlow, we convert the tensor to a numpy array then to a list.
        # This mimics the scalar extraction behavior described in the bug report.
        u0, u1 = a.numpy().tolist()
        
        # Original logic: return a*u0*u1
        with tf.compat.v1.name_scope("computation_scope"):
            return a * u0 * u1

# Test execution
input_tensor = tf.constant([1, 2])
result = func(input_tensor)

# Assertion
# Expected: [1*1, 2*2] = [1, 4]
expected = tf.constant([1, 4])
assert tf.reduce_all(tf.equal(result, expected)).numpy(), "Test case failed: output mismatch"
print("Test passed.")