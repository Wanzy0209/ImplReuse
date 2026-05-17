import torch
import tensorflow as tf

def f(xs):
    # Replacing torch.split with the similar API tf.keras.ops.triu
    return tf.keras.ops.triu(xs)

# Determine device to use. 
# We try to use GPU to mimic the original "cuda" context, 
# but fallback to CPU to ensure the test is runnable in all environments.
device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

with tf.device(device_name):
    xs = tf.random.normal((2, 2))

    # Eager execution
    eager_result = f(xs)
    print("Eager execution successful.")

    # Compiled execution (tf.function is the TensorFlow equivalent of torch.compile)
    # This mimics the scenario where the bug occurred in PyTorch
    compiled_f = tf.function(f)
    compiled_result = compiled_f(xs)
    print("Compiled execution successful.")

    # Verify that the results are consistent
    assert tf.reduce_all(tf.equal(eager_result, compiled_result)).numpy(), "Results differ between eager and compiled modes"

# Outside of device context, verify it still works
compiled_f_outside = tf.function(f)
xs_outside = tf.random.normal((2, 2))
result_outside = compiled_f_outside(xs_outside)
print("Execution outside device context successful.")