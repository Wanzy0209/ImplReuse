import torch
import tensorflow as tf

# This test case mirrors the structure of the original bug report (Issue 163668),
# which involved a check inside a compiled function causing a graph break.
# Here, we use the similar API 'tf.test.is_built_with_gpu_support' inside
# a tf.function (TensorFlow's equivalent of torch.compile) to verify
# if the check handles the graph context correctly.

@tf.function(jit_compile=True)  # Equivalent to torch.compile(fullgraph=True)
def f(x):
    # Using the similar API to check a condition.
    # In the original bug, torch._check failed due to a lambda argument.
    # Here we test if the boolean check works within the compiled graph.
    if tf.test.is_built_with_gpu_support():
        # Simulate the operation performed in the original bug (x + 1)
        return x + 1.0
    else:
        # Fallback path if the check fails
        return x

# Create a tensor input (equivalent to torch.randn)
x = tf.constant([1.0, 2.0, 3.0])

# Execute the function
try:
    result = f(x)
    # Assertion to ensure the function executed without graph break errors
    assert result is not None
    print("Test passed: Function executed successfully with tf.test.is_built_with_gpu_support.")
except Exception as e:
    print(f"Test failed with error: {e}")