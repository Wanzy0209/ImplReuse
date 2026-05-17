import torch
import tensorflow as tf

def f(xs):
    # Using the similar API: tf.keras.ops.tril
    # This replaces xs.split(1, dim=0) from the original PyTorch code
    return tf.keras.ops.tril(xs)

# Compile the function (analogous to torch.compile)
# In TensorFlow, tf.function traces the graph similar to torch.compile
compiled_f = tf.function(f)

# Determine device to use (mimicking the 'cuda' context)
# We default to CPU to ensure the test is runnable on all environments, 
# but the logic applies to device contexts in general.
device = "/CPU:0"
if tf.config.list_physical_devices('GPU'):
    device = "/GPU:0"

print(f"Testing on device: {device}")

# Reproduce the bug scenario: Execution inside a device context
with tf.device(device):
    # Create tensor inside device context
    xs = tf.random.normal((2, 2))

    # 1. Test Eager execution (should work)
    try:
        eager_result = f(xs)
        print("Eager execution successful.")
    except Exception as e:
        print(f"Eager execution failed: {e}")
        raise

    # 2. Test Compiled execution (this is where the PyTorch bug occurred)
    # The original bug was: 'module 'torch._tensor' has no attribute 'split''
    # We check if the similar API fails in the compiled context.
    try:
        compiled_result = compiled_f(xs)
        print("Compiled execution successful.")
        
        # Verify correctness
        assert tf.reduce_all(tf.equal(eager_result, compiled_result)).numpy()
        print("Results match between eager and compiled execution.")
        
    except AttributeError as e:
        # Catching the specific type of error seen in the PyTorch bug
        print(f"Compiled execution failed with AttributeError (similar to PyTorch bug): {e}")
        raise
    except Exception as e:
        print(f"Compiled execution failed with unexpected error: {e}")
        raise