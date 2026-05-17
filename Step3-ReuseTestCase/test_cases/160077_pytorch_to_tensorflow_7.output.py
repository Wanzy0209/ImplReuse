import torch
import tensorflow as tf

def f(xs):
    # Using the similar API: tf.experimental.numpy.triu
    return tf.experimental.numpy.triu(xs)

# Determine device to ensure the test is runnable (mimicking the 'cuda' context)
# Falls back to CPU if GPU is not available, preserving the context logic.
device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

with tf.device(device_name):
    # Create a tensor on the specified device
    xs = tf.random.normal((2, 2))

    # Eager execution (should work)
    eager_result = f(xs)

    # Compiled execution (tf.function is analogous to torch.compile)
    # This mimics the scenario where the original bug occurred
    compiled_f = tf.function(f)
    compiled_result = compiled_f(xs)

    # Verify that the behavior is consistent between eager and compiled modes
    # This assertion checks if the similar API handles the context/compilation correctly
    assert tf.reduce_all(tf.equal(eager_result, compiled_result)).numpy(), \
        "Results differ between eager and compiled execution"

    print("Test passed: Eager and compiled execution produced the same result.")