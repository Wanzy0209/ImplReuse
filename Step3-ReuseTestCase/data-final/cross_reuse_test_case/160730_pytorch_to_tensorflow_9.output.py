import torch
import tensorflow as tf
import numpy as np

def foo(x):
    # Use tf.name_scope to group operations, similar to how torch.compile 
    # groups the graph logic.
    with tf.name_scope("math_computation"):
        t = tf.tan(x)
        # PyTorch's expand is similar to tf.broadcast_to
        e = tf.broadcast_to(t, (31, 51, 1))
        mean_val = tf.reduce_mean(e)
        
        # In TensorFlow, conditional logic on tensor values inside a graph (tf.function)
        # should use tf.cond to ensure compatibility between eager and graph modes.
        # This mimics the original code's control flow.
        def true_branch():
            return tf.subtract(e, e * 0.5)
        
        def false_branch():
            return tf.add(e, e * 0.5)
            
        out1 = tf.cond(mean_val > 0.5, true_branch, false_branch)
        
        return tf.sin(out1)

# Setup input data
np.random.seed(0)
x = np.random.uniform(0, 10, size=(31, 51, 1)).astype(np.float16)
x_tf = tf.constant(x)

# Run in eager mode
eager_res = foo(x_tf)

# Run in compiled/graph mode (tf.function is the TensorFlow equivalent to torch.compile)
# The operations inside foo will be traced within the provided name_scope.
compiled_foo = tf.function(foo)
compiled_res = compiled_foo(x_tf)

# Verify that the results are consistent
try:
    np.testing.assert_allclose(eager_res.numpy(), compiled_res.numpy(), rtol=1e-5, atol=1e-5)
    print("Test passed: Eager and compiled results are consistent.")
except AssertionError as e:
    print(f"Test failed: {e}")