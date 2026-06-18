import torch
import tensorflow as tf
import numpy as np

# The target API: Enable eager execution to ensure the program runs in imperative mode.
# This is the TensorFlow equivalent of the "eager" baseline in the PyTorch bug report.
tf.compat.v1.enable_eager_execution()

def foo(x):
    # Translating PyTorch operations to TensorFlow operations
    t = tf.tan(x)
    # PyTorch's expand is equivalent to tf.broadcast_to
    e = tf.broadcast_to(t, (31, 51, 1))
    
    mean_val = tf.reduce_mean(e)
    
    # In TensorFlow eager mode, this is a standard Python if.
    # When wrapped in tf.function (compiled mode), AutoGraph converts this 
    # to a tf.cond operation, preserving the control flow logic.
    if mean_val > 0.5:
        out1 = tf.subtract(e, e * 0.5)
    else:
        out1 = tf.add(e, e * 0.5)
        
    return tf.sin(out1)

# Reproduce the data setup from the original issue
np.random.seed(0)
x = np.random.uniform(0, 10, size=(31, 51, 1)).astype(np.float16)

# 1. Run in Eager mode (baseline)
eager_res = foo(tf.constant(x))

# 2. Run in Compiled/Graph mode (equivalent to torch.compile)
# tf.function compiles the Python function into a TensorFlow graph
compiled_foo = tf.function(foo)
compile_res = compiled_foo(tf.constant(x))

# 3. Verify that the results match
# This checks if the "compiled" version produces the same result as the "eager" version,
# mirroring the assertion in the original PyTorch bug report.
np.testing.assert_allclose(eager_res.numpy(), compile_res.numpy(), rtol=1e-5, atol=1e-5)

print("Test passed: Eager and Compiled results match.")