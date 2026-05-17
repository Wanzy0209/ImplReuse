import torch
import tensorflow as tf

def f(xs):
    # Original PyTorch API: xs.split(1, dim=0)
    # Adapted TensorFlow API: tf.experimental.numpy.moveaxis
    # We move axis 0 to the last position to verify tensor manipulation
    return tf.experimental.numpy.moveaxis(xs, source=0, destination=-1)

# Determine device to use (GPU if available, otherwise CPU)
# This mimics the 'with torch.device("cuda")' context logic
device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

print(f"Running test on device: {device_name}")

with tf.device(device_name):
    # Create a tensor on the specific device
    # PyTorch: torch.randn(2, 2, device="cuda")
    xs = tf.random.normal((2, 2, 3))

    # 1. Test Eager Execution
    # PyTorch: f(xs)
    print("Testing eager execution...")
    try:
        result_eager = f(xs)
        print(f"Eager result shape: {result_eager.shape}")
        assert result_eager.shape == (2, 3, 2), "Eager execution shape mismatch"
    except Exception as e:
        print(f"Eager execution failed: {e}")

    # 2. Test Compiled Execution
    # PyTorch: torch.compile(f, backend=backend)(xs)
    # TensorFlow equivalent: tf.function(f)
    print("\nTesting compiled execution (tf.function)...")
    try:
        f_compiled = tf.function(f)
        result_compiled = f_compiled(xs)
        
        print(f"Compiled result shape: {result_compiled.shape}")
        
        # Verify correctness
        assert result_compiled.shape == (2, 3, 2), "Compiled execution shape mismatch"
        
        # Verify values match eager execution
        assert tf.reduce_all(tf.equal(result_eager, result_compiled)), "Compiled output differs from eager"
        
        print("Test passed: API works correctly inside device context under compilation.")
        
    except AttributeError as e:
        # Catching the specific type of error reported in the PyTorch issue
        print(f"AttributeError caught (similar to original bug): {e}")
    except Exception as e:
        print(f"Compiled execution failed with unexpected error: {e}")