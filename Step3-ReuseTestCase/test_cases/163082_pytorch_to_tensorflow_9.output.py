import torch
import tensorflow as tf
import numpy as np

# Ensure we are using float32 to match the PyTorch bug report's precision context
dtype = tf.float32

# Define the function wrapped in tf.function (analogous to torch.compile)
@tf.function
def logaddexp2_compiled(x1, x2):
    return tf.experimental.numpy.logaddexp2(x1, x2)

# Define the eager execution function (analogous to vec_norm_without_compile)
def logaddexp2_eager(x1, x2):
    return tf.experimental.numpy.logaddexp2(x1, x2)

# Setup inputs similar to the PyTorch case: specific float values
# PyTorch used [3.799999, 0.0, 0.0]. We map these to x1 and use 0.0 for x2.
x1 = tf.constant([3.799999, 0.0, 0.0], dtype=dtype)
x2 = tf.constant([0.0, 0.0, 0.0], dtype=dtype)

print("Input x1:", [x.item() for x in x1.numpy()])
print("Input x2:", [x.item() for x in x2.numpy()])

# Run compiled version
res_compiled = logaddexp2_compiled(x1, x2)
print("Result (compiled):", [x.item() for x in res_compiled.numpy()])

# Run eager version
res_eager = logaddexp2_eager(x1, x2)
print("Result (eager):", [x.item() for x in res_eager.numpy()])

# Check for precision differences (the core logic of the bug report)
# The PyTorch bug resulted in a norm > 1. Here we check if the compiled
# output deviates from the eager output due to precision issues.
diff = tf.abs(res_compiled - res_eager)
print("Absolute difference:", [x.item() for x in diff.numpy()])

# Assertion to verify behavior (mimicking a test case that would catch the bug)
# We expect the difference to be negligible (close to machine epsilon for float32)
assert tf.reduce_all(diff < 1e-6), f"Significant precision difference detected: {diff.numpy()}"