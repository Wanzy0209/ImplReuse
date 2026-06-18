import torch
import tensorflow as tf

def f(x, y):
    # x.copy_(x.flip(1))
    # In TensorFlow, in-place modification is done via assign on a Variable.
    # tf.reverse is the equivalent of torch.flip.
    x.assign(tf.reverse(x, axis=[1]))
    
    # y = y.sum(dim=1, keepdim=True) + y
    # tf.reduce_sum is the equivalent of torch.sum.
    y = tf.reduce_sum(y, axis=1, keepdims=True) + y
    
    # return x + y
    return x + y

# Check for GPU availability to match the original "cuda" device context
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

with tf.device(device):
    # Initialize inputs
    # x needs to be a Variable to support the in-place assign operation
    x = tf.Variable(tf.random.normal((20, 1024 * 1024)))
    y = tf.random.normal((20, 1024 * 1024))
    
    # Create a copy of x for the second run to ensure independence
    x_copy = tf.Variable(tf.identity(x.value()))
    y_copy = tf.identity(y)

    # Run reference (eager execution)
    ref = f(x, y)

    # Run inside tf.name_scope
    # The prompt asks to adapt the test case to tf.name_scope.
    # Unlike torch.compile, tf.name_scope is a context manager for organizing ops in the graph,
    # not a compiler that performs fusion. We wrap the execution to verify behavior within the scope.
    with tf.name_scope("test_scope"):
        act = f(x_copy, y_copy)

    # Verify results
    # tf.debugging.assert_near is the TensorFlow equivalent of torch.testing.assert_close
    tf.debugging.assert_near(ref, act, message="Numerical difference detected")
    print("Test passed: Results are consistent within tf.name_scope.")